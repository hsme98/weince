import torch
import torch.nn as nn
import torch.nn.functional as F


class NT_Xent(nn.Module):
    """
    NT-Xent (InfoNCE) loss with multiple policies and EVT-driven Weibit:

      policy="pl":
        Standard SimCLR NT-Xent (softmax on cosine/temperature).

      policy="weibit":
        InfoNCE with Weibit logits: -beta_t * log(1 - s), where s is cosine sim.

      policy="soft":
        Global mixture of PL and Weibit:
          L_soft = (1 - soft_lambda) * L_pl + soft_lambda * L_weibit.

      policy="gate":
        Anchor-wise gated mixture of PL and Weibit:
          lambda_i = sigmoid(gate_scale * (log(rho0) - log(rho_i)))
          g_{i,j} = (1-lambda_i) * (s_{i,j}/tau) + lambda_i * (-beta_t log(1-s_{i,j}))
        where rho_i = min_j (1 - s_{i,j}^-) is the per-anchor cap shortfall.

      policy="weibit_shrink":
        Weibit with per-anchor beta_i estimated from each anchor's tail and
        shrunk toward the global beta_t (batch-level estimate):
          beta_i = w_i * beta_hat_i + (1-w_i) * beta_t,
          w_i = (n_eff * R2_i) / (n_eff * R2_i + shrink_c).

      policy="tlambda_select" (NEW):
        Per-anchor *logit* transform that interpolates between PL (InfoNCE)
        and a Weibit-like transform using the map:

            T_\lambda(x) = (1-\lambda) x + \lambda (-log(1-x)).

        Here we apply the transform to cosine similarities (clamped below 1),
        but keep the PL temperature scaling so that \lambda=0 recovers the
        standard NT-Xent logits exactly.

        For an anchor i with per-anchor \lambda_i \in [0,1], we build logits:

            g_{i,j} = (1-\lambda_i) * (s_{i,j}/tau) + \lambda_i * (-log(1-s_{i,j}))

        so that exp(g_{i,j}) is a geometric mixture of InfoNCE weights and a
        Weibit-like factor (1-s)^{-\lambda_i}. We estimate \lambda_i using the
        same per-anchor tail evidence (AIC and cap-proximity) used by
        soft_select/hard_select, but we do **not** modify those policies.

    Additionally, an optional Weibull top-M regularizer can be added:

      L_total = L_policy + weib_lambda * L_weib_topM

    where L_weib_topM is a Weibit InfoNCE over [positive | top-M hardest negatives]
    per anchor.

    The Weibit shape parameter beta is either:
      - fixed (weib_beta) if auto_weib_beta=False, or
      - estimated per batch from EVT diagnostics on the negative tail if
        auto_weib_beta=True (beta_t, with EMA and clamping).
    """

    def __init__(
        self,
        batch_size: int,
        temperature: float,
        world_size: int = 1,
        policy: str = "pl",           # "pl", "weibit", "soft", "gate", "weibit_shrink", "hard_select", "soft_select", "tlambda_select"
        weib_beta: float = 4.0,       # initial / fixed β if auto_weib_beta=False
        soft_lambda: float = 0.5,     # mix weight for "soft" policy
        use_weib_topm: bool = False,
        weib_lambda: float = 0.0,     # weight for top-M regularizer
        weib_top_m: int = 16,         # number of hardest negatives per anchor
        eps: float = 1e-6,
        pl_lambda: float = 0.0,         # weight for pl top-M regularizer for pl ablation study
        pl_top_m: int = 16,           # number of hardest negatives per anchor for pl ablation study
        use_pl_topm: bool = False, # if True, add pl top-M regularizer for pl ablation study

        # EVT / β diagnostics
        auto_weib_beta: bool = False, # if True, estimate β_t from tail each batch
        beta_min: float = 0.5,
        beta_max: float = 16.0,
        tail_frac: float = 0.05,      # fraction of negative tail used for regression
        min_tail_points: int = 1024,  # minimum #tail points
        beta_ema_momentum: float = 0.9,
        r2_min: float = 0.85,         # minimum R^2 to trust a new β̂

        # (4) Anchor-wise gating hyperparameters (policy="gate")
        gate_rho0: float = 0.05,
        gate_scale: float = 5.0,
        gate_r2_min: float = 0.0,

        # (5) Per-anchor beta shrinkage hyperparameters (policy="weibit_shrink")
        anchor_tail_frac: float = 0.10,
        anchor_min_tail_points: int = 32,
        anchor_r2_min: float = 0.0,
        shrink_c: float = 100.0,

        # (6) Hard per-anchor selection between PL vs Weibit (policy="hard_select")
        # Selection is binary: lambda_i \in {0,1} per anchor.
        # We select Weibit when (i) the anchor is "near the cap" (small rho_i)
        # and (ii) a Weibull tail fit beats a Gumbel-fast tail fit by AIC margin.
        select_q: float = 0.2,                 # fraction of hardest anchors (by rho) eligible for Weibit
        select_aic_margin: float = 2.0,        # AIC_G - AIC_W must exceed this margin to choose Weibit
        select_tail_frac: float = 0.10,        # per-anchor tail fraction used for AIC fits
        select_min_tail_points: int = 32,      # minimum tail points used for AIC fits
        select_weib_top_m: int = 16,           # top-M negatives used for Weibit loss when selected
        select_kappa_rho: float = 10.0,           # softness for rho-based gate (larger => harder)
        select_kappa_aic: float = 1.0,            # softness for AIC-based gate (larger => harder)
    ):
        super(NT_Xent, self).__init__()

        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.policy = policy
        self.weib_beta_init = weib_beta
        self.soft_lambda = soft_lambda

        # Weibull top-M regularizer config
        self.use_weib_topm = use_weib_topm
        self.weib_lambda = weib_lambda
        self.weib_top_m = weib_top_m
        self.eps = eps

        # PL top-M regularizer config
        self.use_pl_topm = use_pl_topm
        self.pl_lambda = pl_lambda
        self.pl_top_m = pl_top_m

        # EVT / β diagnostics config
        self.auto_weib_beta = auto_weib_beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.tail_frac = tail_frac
        self.min_tail_points = min_tail_points
        self.beta_ema_momentum = beta_ema_momentum
        self.r2_min = r2_min

        # (4) gate policy
        self.gate_rho0 = gate_rho0
        self.gate_scale = gate_scale
        self.gate_r2_min = gate_r2_min

        # (5) per-anchor shrink
        self.anchor_tail_frac = anchor_tail_frac
        self.anchor_min_tail_points = anchor_min_tail_points
        self.anchor_r2_min = anchor_r2_min
        self.shrink_c = shrink_c

        # (6) hard_select
        self.select_q = float(select_q)
        self.select_aic_margin = float(select_aic_margin)
        self.select_tail_frac = float(select_tail_frac)
        self.select_min_tail_points = int(select_min_tail_points)
        self.select_weib_top_m = int(select_weib_top_m)
        self.select_kappa_rho = float(select_kappa_rho)
        self.select_kappa_aic = float(select_kappa_aic)

        # Running β_t (EMA) and last diagnostics
        self.register_buffer("beta_ema", torch.tensor(float(weib_beta)))
        self.last_beta_hat = None
        self.last_beta_t = float(weib_beta)
        self.last_R2 = None
        self.last_rho = None
        self.last_gate_mean = None
        self.last_beta_anchor_mean = None
        self.last_beta_anchor_std = None
        self.last_beta_anchor_r2_mean = None

        # hard_select diagnostics
        self.last_select_frac = None
        self.last_select_rho0 = None
        self.last_select_dAIC_mean = None
        self.last_select_beta_hat_mean = None

        # soft-select diagnostics (optional)
        self.last_select_lam_rho_mean = None
        self.last_select_lam_aic_mean = None

        # SimCLR-style negatives mask: shape (2N, 2N)
        n_samples = batch_size * world_size
        self.mask = self._get_correlated_mask(n_samples).type(torch.bool)

        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def _get_correlated_mask(self, n_samples: int) -> torch.Tensor:
        """
        Build a mask of shape (2N,2N) that excludes:
          - self pairs (i,i)
          - positive pairs (i, i+N) and (i+N, i)
        where N = n_samples.
        """
        N = n_samples
        N2 = 2 * N
        mask = torch.ones((N2, N2), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(N):
            j = i + N
            mask[i, j] = False
            mask[j, i] = False
        return mask

    @torch.no_grad()
    def _estimate_weib_beta_from_tail(self, sim_cos: torch.Tensor):
        """
        EVT diagnostics: estimate β̂, R², and rho from the global negative tail.

        sim_cos: (2N,2N) cosine similarities BEFORE temperature.

        Procedure:
          - Collect all negative similarities (i,j) excluding self and positives.
          - Convert to shortfalls δ = 1 - s.
          - Take the smallest tail_frac fraction (at least min_tail_points).
          - Fit log F(δ) ≈ log c + β log δ (global tail).
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # mask negatives
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos[mask]  # 1D, all negatives
        if sim_neg.numel() == 0:
            return float(self.beta_ema.item()), 0.0, None

        delta = (1.0 - sim_neg).clamp_min(self.eps)  # shortfalls
        K = delta.numel()
        k_tail = int(max(self.min_tail_points, self.tail_frac * K))
        k_tail = min(k_tail, K)
        if k_tail < 10:
            # not enough tail points
            return float(self.beta_ema.item()), 0.0, float(delta.min().item())

        # Take smallest k_tail shortfalls: global tail
        delta_tail, _ = torch.topk(delta, k=k_tail, largest=False, sorted=True)  # ascending
        # Empirical CDF values
        k = torch.arange(1, k_tail + 1, device=device, dtype=delta_tail.dtype)
        Fhat = k / (k_tail + 1.0)

        x = torch.log(delta_tail)
        y = torch.log(Fhat)

        # burn a few points to reduce discretization noise
        burn = max(1, k_tail // 20)
        xb = x[burn:]
        yb = y[burn:]
        if xb.numel() < 2:
            return float(self.beta_ema.item()), 0.0, float(delta.min().item())

        xmean = xb.mean()
        ymean = yb.mean()
        xc = xb - xmean
        yc = yb - ymean
        den = (xc * xc).sum().clamp_min(1e-12)
        num = (xc * yc).sum()
        beta_hat = (num / den).clamp_min(0.0)

        # R^2
        yhat = (num / den) * xc + ymean
        ss_res = ((yb - yhat) ** 2).sum()
        ss_tot = ((yb - ymean) ** 2).sum().clamp_min(1e-12)
        R2 = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)

        rho = float(delta.min().item())
        return float(beta_hat.item()), float(R2.item()), rho

    def _get_batch_beta(self, sim_cos: torch.Tensor) -> float:
        """
        Decide which β to use this batch:
          - if auto_weib_beta=False: use fixed weib_beta_init
          - else: estimate β̂ from tail, trust only if R² >= r2_min, update EMA,
            clamp to [beta_min, beta_max], and return β_t.
        """
        if not self.auto_weib_beta:
            self.last_beta_hat = None
            self.last_R2 = None
            self.last_rho = None
            self.last_beta_t = float(self.weib_beta_init)
            return float(self.weib_beta_init)

        beta_hat, R2, rho = self._estimate_weib_beta_from_tail(sim_cos)
        # Fallback: if fit is bad or beta_hat is zero, keep previous EMA
        beta_new = float(self.beta_ema.item())
        if R2 >= self.r2_min and beta_hat > 0.0:
            beta_clamped = max(self.beta_min, min(self.beta_max, beta_hat))
            m = self.beta_ema_momentum
            beta_new = m * float(self.beta_ema.item()) + (1.0 - m) * beta_clamped
            self.beta_ema.data.copy_(torch.tensor(beta_new, device=self.beta_ema.device))

        # store diagnostics
        self.last_beta_hat = beta_hat
        self.last_R2 = R2
        self.last_rho = rho
        self.last_beta_t = beta_new
        return beta_new

    def _build_pl_logits_and_labels(
        self, sim_cos: torch.Tensor
    ):
        """
        Build PL (standard NT-Xent) logits and labels.

        sim_cos: (2N,2N) cosine similarities BEFORE temperature scaling.
        """
        device = sim_cos.device
        sim = sim_cos / self.temperature

        N2 = sim.size(0)
        N = N2 // 2

        # positives: sim(i, i+N) and sim(i+N, i)
        sim_i_j = torch.diag(sim, N)       # (N,)
        sim_j_i = torch.diag(sim, -N)      # (N,)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0).view(N2, 1)  # (2N,1)

        # negatives: all others according to mask
        mask = self.mask.to(device)
        negatives = sim[mask].view(N2, -1)  # (2N, 2N-2)

        logits = torch.cat([positives, negatives], dim=1)  # (2N, 1+2N-2)
        labels = torch.zeros(N2, dtype=torch.long, device=device)
        return logits, labels

    def _build_weibit_logits_and_labels(
        self, sim_cos: torch.Tensor, beta_t: float
    ):
        """
        Build Weibit logits and labels using batch-dependent beta_t.

        sim_cos: (2N,2N) cosine similarities BEFORE temperature.
        beta_t:  scalar β_t to use this batch.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # positives
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        # negatives: mask out self and positives
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False
        s_neg = sim_cos[mask].view(N2, -1)          # (2N, 2N-2)

        # Weibit logits: -β_t log(1 - s), clamp at 1-eps
        eps = self.eps
        s_pos_clamped = s_pos.clamp(max=1.0 - eps)
        s_neg_clamped = s_neg.clamp(max=1.0 - eps)

        beta = float(beta_t)
        pos_logits = -beta * torch.log(1.0 - s_pos_clamped)   # (2N,1)
        neg_logits = -beta * torch.log(1.0 - s_neg_clamped)   # (2N,2N-2)
        logits = torch.cat([pos_logits, neg_logits], dim=1)   # (2N, 1+2N-2)
        labels = torch.zeros(N2, dtype=torch.long, device=device)
        return logits, labels

    @torch.no_grad()
    def _anchorwise_gate_lambda(self, sim_cos: torch.Tensor) -> torch.Tensor:
        """Compute per-anchor gate weights lambda_i in [0,1].

        This implements suggestion (4): detect whether each anchor is operating
        near the cap and (optionally) whether the batch tail fit is reliable.

        lambda_i = sigmoid(gate_scale * (log(gate_rho0) - log(rho_i)))
        where rho_i = min_j (1 - s_{i,j}^-) over negatives j.

        If the batch-level Weibull tail regression quality is poor
        (last_R2 < gate_r2_min), we disable gating and return zeros.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # negatives mask (exclude self and positives)
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        s_neg = sim_cos[mask].view(N2, -1)  # (2N, 2N-2)
        rho_i = (1.0 - s_neg).clamp_min(self.eps).min(dim=1).values  # (2N,)

        # gate off if tail fit is unreliable
        if (self.last_R2 is None) or (float(self.last_R2) < float(self.gate_r2_min)):
            lam = torch.zeros_like(rho_i)
        else:
            rho0 = torch.tensor(max(float(self.gate_rho0), self.eps), device=device, dtype=rho_i.dtype)
            lam = torch.sigmoid(self.gate_scale * (torch.log(rho0) - torch.log(rho_i)))

        self.last_gate_mean = float(lam.mean().item())
        return lam

    def _build_gate_logits_and_labels(self, sim_cos: torch.Tensor, beta_t: float):
        """Anchor-wise gated mixture of PL and Weibit logits (policy="gate").

        For each anchor i we compute a gate lambda_i and mix logits:

          g_{i,j} = (1-lambda_i) * (s_{i,j}/tau) + lambda_i * (-beta_t log(1-s_{i,j}))

        Output format matches other builders: logits of shape (2N, 1+2N-2)
        with the positive at column 0.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # positives / negatives in cosine space
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False
        s_neg = sim_cos[mask].view(N2, -1)          # (2N,2N-2)

        # gate weights per anchor
        lam = self._anchorwise_gate_lambda(sim_cos).view(N2, 1)  # (2N,1)

        # PL logits
        pl_pos = s_pos / self.temperature
        pl_neg = s_neg / self.temperature

        # Weibit logits
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        s_neg_c = s_neg.clamp(max=1.0 - eps)
        beta = float(beta_t)
        wb_pos = -beta * torch.log(1.0 - s_pos_c)
        wb_neg = -beta * torch.log(1.0 - s_neg_c)

        # mix per anchor
        pos_logits = (1.0 - lam) * pl_pos + lam * wb_pos
        neg_logits = (1.0 - lam) * pl_neg + lam * wb_neg

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(N2, dtype=torch.long, device=device)
        return logits, labels

    @torch.no_grad()
    def _estimate_anchor_betas_from_tail(self, s_neg: torch.Tensor):
        """Estimate per-anchor beta_hat_i and R2_i from each anchor's negative tail.

        s_neg: (2N, Kneg) cosine similarities of negatives for each anchor.
        Returns:
          beta_hat: (2N,)
          R2:       (2N,)
          n_eff:    int (number of points used in regression)
        """
        device = s_neg.device
        N2, Kneg = s_neg.shape

        delta = (1.0 - s_neg).clamp_min(self.eps)  # shortfalls
        k_tail = int(max(self.anchor_min_tail_points, self.anchor_tail_frac * Kneg))
        k_tail = min(k_tail, Kneg)
        if k_tail < 10:
            beta_hat = torch.zeros(N2, device=device, dtype=delta.dtype)
            R2 = torch.zeros_like(beta_hat)
            return beta_hat, R2, 0

        # smallest k_tail deltas per anchor
        delta_tail, _ = torch.topk(delta, k=k_tail, dim=1, largest=False, sorted=True)  # (2N,k_tail)
        k = torch.arange(1, k_tail + 1, device=device, dtype=delta_tail.dtype)
        Fhat = (k / (k_tail + 1.0)).clamp_min(self.eps)  # (k_tail,)

        x = torch.log(delta_tail)                # (2N,k_tail)
        y = torch.log(Fhat).unsqueeze(0)         # (1,k_tail) broadcast

        burn = max(1, k_tail // 20)
        xb = x[:, burn:]
        yb = y[:, burn:]
        n_eff = xb.size(1)
        if n_eff < 2:
            beta_hat = torch.zeros(N2, device=device, dtype=delta.dtype)
            R2 = torch.zeros_like(beta_hat)
            return beta_hat, R2, 0

        xmean = xb.mean(dim=1, keepdim=True)
        ymean = yb.mean()  # scalar
        xc = xb - xmean
        yc = yb - ymean
        den = (xc * xc).sum(dim=1).clamp_min(1e-12)
        num = (xc * yc).sum(dim=1)
        beta_hat = (num / den).clamp_min(0.0)

        # R^2 per anchor
        slope = (num / den).unsqueeze(1)
        yhat = slope * xc + ymean
        ss_res = ((yb - yhat) ** 2).sum(dim=1)
        ss_tot = ((yb - ymean) ** 2).sum().clamp_min(1e-12)
        R2 = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        return beta_hat, R2, int(n_eff)

    def _build_weibit_shrink_logits_and_labels(self, sim_cos: torch.Tensor, beta_t: float):
        """Weibit logits with per-anchor beta shrinkage (policy="weibit_shrink").

        Steps:
          1) Build per-anchor negative matrix s_neg.
          2) Estimate per-anchor beta_hat_i and R2_i from each anchor's tail.
          3) Shrink toward global beta_t using w_i = (n_eff*R2_i)/(n_eff*R2_i + shrink_c).
          4) Use beta_i to form Weibit logits -beta_i log(1-s).
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False
        s_neg = sim_cos[mask].view(N2, -1)          # (2N,2N-2)

        # per-anchor beta hats
        beta_hat, R2, n_eff = self._estimate_anchor_betas_from_tail(s_neg)

        # shrink weights
        if n_eff <= 0:
            w = torch.zeros_like(beta_hat)
        else:
            nR = float(n_eff) * R2
            w = nR / (nR + float(self.shrink_c))

        # drop anchors with poor fit
        if self.anchor_r2_min > 0.0:
            w = w * (R2 >= float(self.anchor_r2_min)).to(w.dtype)
        w = w * (beta_hat > 0.0).to(w.dtype)

        beta_global = torch.tensor(float(beta_t), device=device, dtype=beta_hat.dtype)
        beta_i = w * beta_hat + (1.0 - w) * beta_global
        beta_i = beta_i.clamp(min=float(self.beta_min), max=float(self.beta_max))

        # diagnostics
        self.last_beta_anchor_mean = float(beta_i.mean().item())
        self.last_beta_anchor_std = float(beta_i.std(unbiased=False).item())
        self.last_beta_anchor_r2_mean = float(R2.mean().item())

        # Weibit logits with per-anchor beta
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        s_neg_c = s_neg.clamp(max=1.0 - eps)
        beta_row = beta_i.view(N2, 1)
        pos_logits = -beta_row * torch.log(1.0 - s_pos_c)
        neg_logits = -beta_row * torch.log(1.0 - s_neg_c)

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(N2, dtype=torch.long, device=device)
        return logits, labels

    # ----------------------- hard_select (binary per-anchor switch) -----------------------

    @torch.no_grad()
    def _hard_select_decision(self, sim_cos: torch.Tensor):
        """Compute binary per-anchor selector \lambda_i for policy="hard_select".

        We treat each anchor row i as having a set of negative similarities s_{i,j}.
        Define shortfalls \delta_{i,j} = 1 - s_{i,j}. The smallest shortfalls correspond
        to the *hardest* negatives (closest to the cap).

        We compute two simple tail fits on the smallest k_tail shortfalls:
          (W) Weibull/power tail:        log F(\delta) \approx a + \beta log \delta
          (G) Gumbel-fast tail (proxy):  log(-log F(\delta)) \approx b - \kappa log \delta

        Using an AIC computed from the OLS residuals, we pick Weibull if it beats
        the Gumbel proxy by at least `select_aic_margin`, and only for anchors
        that are among the hardest `select_q` fraction by cap-proximity.

        Returns:
          lam_bin: (2N,) float in {0,1}
          topm_idx: (2N, M) long indices in [0,2N) selecting top-M negatives per anchor
          beta_hat: (2N,) estimated Weibull exponent (slope), clamped >=0
          R2_w:     (2N,) Weibull-fit R^2 in [0,1]
          dAIC:     (2N,) AIC_G - AIC_W (positive favors Weibull)
          rho_i:    (2N,) cap proximity proxy rho_i = 1 - max_j s_{i,j}^-
          rho0:     scalar pivot (q-quantile of rho)
          n_eff:    int number of tail points used
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # Work on detached similarities for selection / diagnostics.
        sim_det = sim_cos.detach()

        # Mask negatives (exclude self and positives)
        mask = torch.ones_like(sim_det, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_det.masked_fill(~mask, -1e9)  # (2N,2N)
        Kneg = N2 - 2

        # --- choose tail size for AIC fits ---
        k_tail = int(max(self.select_min_tail_points, round(self.select_tail_frac * Kneg)))
        # Keep k_tail within [1, Kneg]. Some callers (e.g., unit tests) may use
        # very small batch sizes where Kneg < 10.
        k_tail = min(k_tail, Kneg)
        k_tail = max(10, k_tail)
        k_tail = min(k_tail, Kneg)

        # top-k tail values/indices (sorted by similarity desc => shortfalls asc)
        top_vals_tail, top_idx_tail = torch.topk(
            sim_neg, k=k_tail, dim=1, largest=True, sorted=True
        )  # (2N,k_tail)

        # cap proximity proxy rho_i = 1 - max negative similarity
        rho_i = (1.0 - top_vals_tail[:, 0]).clamp_min(self.eps)  # (2N,)

        # --- top-M indices for Weibit loss when selected ---
        M = int(self.select_weib_top_m)
        M = min(M, Kneg)
        M = max(1, M)
        if k_tail >= M:
            topm_idx = top_idx_tail[:, :M]
        else:
            # (should not happen due to k_tail>=10), but keep safe.
            _, topm_idx = torch.topk(sim_neg, k=M, dim=1, largest=True, sorted=False)

        # --- build tail shortfalls delta ---
        delta_tail = (1.0 - top_vals_tail).clamp_min(self.eps)  # (2N,k_tail)
        x = torch.log(delta_tail)  # (2N,k_tail)

        k = torch.arange(1, k_tail + 1, device=device, dtype=x.dtype)
        Fhat = (k / (k_tail + 1.0)).clamp_min(self.eps)

        burn = max(1, k_tail // 20)
        xb = x[:, burn:]
        n_eff = xb.size(1)
        if n_eff < 2:
            lam_bin = torch.zeros((N2,), device=device, dtype=x.dtype)
            beta_hat = torch.zeros_like(lam_bin)
            R2_w = torch.zeros_like(lam_bin)
            dAIC = torch.zeros_like(lam_bin)
            rho0 = float(rho_i.median().item())
            return lam_bin, topm_idx, beta_hat, R2_w, dAIC, rho_i, rho0, int(n_eff)

        Fhat_b = Fhat[burn:]
        yW = torch.log(Fhat_b)                 # (n_eff,)
        yG = torch.log(-torch.log(Fhat_b))     # (n_eff,)

        # Helper: per-row OLS slope and SSE for y ~ a + b x
        def _ols_slope_sse(x_rows: torch.Tensor, y_vec: torch.Tensor):
            xmean = x_rows.mean(dim=1, keepdim=True)
            ymean = y_vec.mean()
            xc = x_rows - xmean
            yc = y_vec - ymean
            den = (xc * xc).sum(dim=1).clamp_min(1e-12)
            num = (xc * yc.unsqueeze(0)).sum(dim=1)
            slope = num / den
            yhat = slope.unsqueeze(1) * xc + ymean
            sse = ((y_vec.unsqueeze(0) - yhat) ** 2).sum(dim=1).clamp_min(1e-12)
            return slope, sse

        slopeW, sseW = _ols_slope_sse(xb, yW)
        slopeG, sseG = _ols_slope_sse(xb, yG)

        # R^2 for Weibull fit
        ss_tot_W = ((yW - yW.mean()) ** 2).sum().clamp_min(1e-12)
        R2_w = (1.0 - sseW / ss_tot_W).clamp(0.0, 1.0)

        # Penalize nonsensical slopes (avoid selecting due to noise)
        # Weibull requires beta>0; Gumbel proxy expects slope<0.
        badW = slopeW <= 0.0
        badG = slopeG >= 0.0
        sseW = torch.where(badW, sseW + 1e6, sseW)
        sseG = torch.where(badG, sseG + 1e6, sseG)

        n = float(n_eff)
        # Gaussian-regression AIC up to additive constants
        aicW = n * torch.log(sseW / n) + 2.0 * 2.0
        aicG = n * torch.log(sseG / n) + 2.0 * 2.0
        dAIC = aicG - aicW

        # --- cap proximity quantile pivot rho0 ---
        rho_sorted, _ = torch.sort(rho_i)
        q = float(self.select_q)
        q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
        kq = int(round(q * (rho_sorted.numel() - 1)))
        rho0 = rho_sorted[kq]
        near_cap = rho_i <= rho0

        # --- final binary decision ---
        choose_weib = dAIC >= float(self.select_aic_margin)
        lam_bin = (near_cap & choose_weib).to(x.dtype)

        beta_hat = slopeW.clamp_min(0.0)

        # diagnostics
        self.last_select_frac = float(lam_bin.mean().item())
        self.last_select_rho0 = float(rho0.item())
        self.last_select_dAIC_mean = float(dAIC.mean().item())
        self.last_select_beta_hat_mean = float(beta_hat.mean().item())

        return lam_bin, topm_idx, beta_hat, R2_w, dAIC, rho_i, float(rho0.item()), int(n_eff)

    def _hard_select_losses(self, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
        """Compute the hard_select objective as a mean over anchors.

        Each anchor i chooses either:
          - PL / standard InfoNCE loss over the full negative set
          - Weibit loss over {positive + top-M hardest negatives}
        based on a binary selector \lambda_i from `_hard_select_decision`.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        # PL per-anchor losses (full set)
        logits_pl, labels_pl = self._build_pl_logits_and_labels(sim_cos)
        loss_pl_row = F.cross_entropy(logits_pl, labels_pl, reduction="none")  # (2N,)

        # Hard selection decision (no grad)
        lam, topm_idx, beta_hat, R2_w, _dAIC, _rho, _rho0, n_eff = self._hard_select_decision(sim_cos)
        lam = lam.detach().to(device=device, dtype=loss_pl_row.dtype)  # (2N,)

        # Weibit per-anchor losses (positive + top-M)
        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)
        s_neg_topm = sim_cos.gather(1, topm_idx)    # (2N,M)

        # Per-anchor beta: shrink toward global beta_t using existing shrink_c/anchor_r2_min
        if n_eff <= 0:
            w = torch.zeros_like(beta_hat)
        else:
            nR = float(n_eff) * R2_w
            w = nR / (nR + float(self.shrink_c))
        if self.anchor_r2_min > 0.0:
            w = w * (R2_w >= float(self.anchor_r2_min)).to(w.dtype)
        w = w * (beta_hat > 0.0).to(w.dtype)

        beta_global = torch.tensor(float(beta_t), device=device, dtype=beta_hat.dtype)
        beta_i = (w * beta_hat + (1.0 - w) * beta_global).clamp(
            min=float(self.beta_min), max=float(self.beta_max)
        )

        # Weibit logits
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        s_neg_c = s_neg_topm.clamp(max=1.0 - eps)
        beta_row = beta_i.to(device=device, dtype=s_pos_c.dtype).view(N2, 1)
        pos_logits = -beta_row * torch.log(1.0 - s_pos_c)
        neg_logits = -beta_row * torch.log(1.0 - s_neg_c)
        logits_w = torch.cat([pos_logits, neg_logits], dim=1)  # (2N,1+M)
        labels_w = torch.zeros(N2, dtype=torch.long, device=device)
        loss_w_row = F.cross_entropy(logits_w, labels_w, reduction="none")

        # Final hard selection
        loss_row = (1.0 - lam) * loss_pl_row + lam * loss_w_row
        return loss_row.mean()



    # ----------------------- soft_select (per-anchor soft switch) -----------------------

    @torch.no_grad()
    def _soft_select_lambda(self, dAIC: torch.Tensor, rho_i: torch.Tensor, rho0: float, mode: str = "both") -> torch.Tensor:
        """Compute soft per-anchor selector \lambda_i \in [0,1].

        This is a soft relaxation of `hard_select`. It uses two confidence signals:

          (1) Near-cap / hardness (rho):
                rho_i = 1 - max_j s_{i,j}^-  (small rho => near cap)
              lam_rho = sigmoid(kappa_rho * (log(rho0) - log(rho_i)))
              where rho0 is the `select_q`-quantile pivot.

          (2) Tail-shape evidence (AIC):
              dAIC_i = AIC_G - AIC_W  (positive favors Weibull)
              lam_aic = sigmoid(kappa_aic * (dAIC_i - select_aic_margin))

        Modes:
          - mode="both": lam = lam_rho * lam_aic
          - mode="rho" : lam = lam_rho
          - mode="aic" : lam = lam_aic
        """
        device = rho_i.device
        dtype = rho_i.dtype
        eps = float(self.eps)

        rho0_t = torch.tensor(max(float(rho0), eps), device=device, dtype=dtype)
        lam_rho = torch.sigmoid(self.select_kappa_rho * (torch.log(rho0_t) - torch.log(rho_i.clamp_min(eps))))

        lam_aic = torch.sigmoid(self.select_kappa_aic * (dAIC - float(self.select_aic_margin)))

        mode = str(mode).lower()
        if mode == "rho":
            lam = lam_rho
        elif mode == "aic":
            lam = lam_aic
        else:
            lam = lam_rho * lam_aic

        # diagnostics
        self.last_select_lam_rho_mean = float(lam_rho.mean().item())
        self.last_select_lam_aic_mean = float(lam_aic.mean().item())
        self.last_select_frac = float(lam.mean().item())
        return lam

    def _soft_select_losses(self, sim_cos: torch.Tensor, beta_t: float, mode: str = "both") -> torch.Tensor:
        """Compute soft-select objective as a mean over anchors.

        Soft-select mixes per-anchor losses:
          loss_i = (1-lam_i) * L_PL_i + lam_i * L_WeibTopM_i,
        where lam_i \in [0,1] is computed by `_soft_select_lambda`.

        `mode` controls which confidence signals are used ("both"/"rho"/"aic").
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        # PL per-anchor losses (full set)
        logits_pl, labels_pl = self._build_pl_logits_and_labels(sim_cos)
        loss_pl_row = F.cross_entropy(logits_pl, labels_pl, reduction="none")  # (2N,)

        # Reuse the tail fits from hard_select_decision (no-grad)
        _lam_bin, topm_idx, beta_hat, R2_w, dAIC, rho_i, rho0, n_eff = self._hard_select_decision(sim_cos)

        # Soft selector lambda_i
        lam = self._soft_select_lambda(
            dAIC.to(device=device, dtype=loss_pl_row.dtype),
            rho_i.to(device=device, dtype=loss_pl_row.dtype),
            rho0,
            mode=mode,
        ).detach()

        # Weibit per-anchor losses (positive + top-M)
        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)
        s_neg_topm = sim_cos.gather(1, topm_idx)    # (2N,M)

        # Per-anchor beta: shrink toward global beta_t (same as hard_select)
        if n_eff <= 0:
            w = torch.zeros_like(beta_hat)
        else:
            nR = float(n_eff) * R2_w
            w = nR / (nR + float(self.shrink_c))
        if self.anchor_r2_min > 0.0:
            w = w * (R2_w >= float(self.anchor_r2_min)).to(w.dtype)
        w = w * (beta_hat > 0.0).to(w.dtype)

        beta_global = torch.tensor(float(beta_t), device=device, dtype=beta_hat.dtype)
        beta_i = (w * beta_hat + (1.0 - w) * beta_global).clamp(
            min=float(self.beta_min), max=float(self.beta_max)
        )

        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        s_neg_c = s_neg_topm.clamp(max=1.0 - eps)
        beta_row = beta_i.to(device=device, dtype=s_pos_c.dtype).view(N2, 1)
        pos_logits = -beta_row * torch.log(1.0 - s_pos_c)
        neg_logits = -beta_row * torch.log(1.0 - s_neg_c)
        logits_w = torch.cat([pos_logits, neg_logits], dim=1)  # (2N,1+M)
        labels_w = torch.zeros(N2, dtype=torch.long, device=device)
        loss_w_row = F.cross_entropy(logits_w, labels_w, reduction="none")

        loss_row = (1.0 - lam) * loss_pl_row + lam * loss_w_row
        return loss_row.mean()


    # ----------------------- tlambda_select (per-anchor logit transform) -----------------------

    def _build_tlambda_logits_and_labels(self, sim_cos: torch.Tensor, mode: str = "both"):
        """Build logits for policy="tlambda_select".

        We reuse the same per-anchor diagnostics from `_hard_select_decision`
        (AIC evidence + cap-proximity) to compute a *soft* selector
        \lambda_i \in [0,1] via `_soft_select_lambda`.

        Then we apply the per-anchor transform (logit-level interpolation):

            g_{i,j} = (1-\lambda_i) * (s_{i,j}/tau) + \lambda_i * (-log(1 - s_{i,j}))

        where s_{i,j} is cosine similarity (clamped below 1 for numerical stability).

        This differs from `soft_select`, which mixes *losses*; here we mix the
        *logits*, i.e. the mixture is "exponential" after softmax.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        # Pos/neg indices
        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # Extract positives and negatives (cosine space)
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False
        s_neg = sim_cos[mask].view(N2, -1)          # (2N, 2N-2)

        # Compute per-anchor lambda_i (no-grad)
        _lam_bin, _topm_idx, _beta_hat, _R2_w, dAIC, rho_i, rho0, _n_eff = self._hard_select_decision(sim_cos)
        lam = self._soft_select_lambda(
            dAIC.to(device=device, dtype=sim_cos.dtype),
            rho_i.to(device=device, dtype=sim_cos.dtype),
            rho0,
            mode=mode,
        ).detach().view(N2, 1)

        # PL logits
        pl_pos = s_pos / self.temperature
        pl_neg = s_neg / self.temperature

        # Weibit-like logits with beta=1 (beta is effectively \lambda in the mixture)
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        s_neg_c = s_neg.clamp(max=1.0 - eps)
        wb_pos = -torch.log(1.0 - s_pos_c)
        wb_neg = -torch.log(1.0 - s_neg_c)

        # Apply T_lambda per anchor
        pos_logits = (1.0 - lam) * pl_pos + lam * wb_pos
        neg_logits = (1.0 - lam) * pl_neg + lam * wb_neg

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(N2, dtype=torch.long, device=device)
        return logits, labels
    def _weibull_topm_loss(self, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
        """
        Weibit top-M loss on the raw cosine similarity matrix, with batch β_t.

        sim_cos: (2N,2N) cosine similarity BEFORE temperature scaling.
        beta_t: scalar β_t to use this batch.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # mask out self and positives to get negatives
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos.masked_fill(~mask, -1e9)
        K = sim_neg.size(1)
        top_m = min(self.weib_top_m, max(1, K - 1))

        # top-M hardest negatives
        top_vals, _ = torch.topk(sim_neg, k=top_m, dim=1, largest=True, sorted=False)  # (2N,M)

        # positive similarities
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        eps = self.eps
        s_pos_clamped = s_pos.clamp(max=1.0 - eps)
        s_neg_clamped = top_vals.clamp(max=1.0 - eps)

        beta = float(beta_t)
        pos_logits = -beta * torch.log(1.0 - s_pos_clamped)      # (2N,1)
        neg_logits = -beta * torch.log(1.0 - s_neg_clamped)      # (2N,M)
        logits = torch.cat([pos_logits, neg_logits], dim=1)      # (2N,1+M)

        labels = torch.zeros(N2, dtype=torch.long, device=device)
        loss_weib = F.cross_entropy(logits, labels, reduction="mean")
        return loss_weib
    
    def _pl_topm_loss(self, sim_cos: torch.Tensor) -> torch.Tensor:
        """
        Used for ablation study.
        Weibit top-M loss on the raw cosine similarity matrix, with batch β_t.

        sim_cos: (2N,2N) cosine similarity BEFORE temperature scaling.
        beta_t: scalar β_t to use this batch.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # mask out self and positives to get negatives
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos.masked_fill(~mask, -1e9)
        K = sim_neg.size(1)
        top_m = min(self.pl_top_m, max(1, K - 1))

        # top-M hardest negatives
        top_vals, _ = torch.topk(sim_neg, k=top_m, dim=1, largest=True, sorted=False)  # (2N,M)

        # positive similarities
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        logits = torch.cat([s_pos/self.temperature, top_vals/self.temperature], dim=1)      # (2N,1+M)

        labels = torch.zeros(N2, dtype=torch.long, device=device)
        loss_weib = self.ce(logits, labels)
        return loss_weib

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j: (batch_size*world_size, d)

        In distributed training, you should all-gather z_i, z_j before
        calling forward so that their batch dimension is batch_size*world_size.
        """
        device = z_i.device
        N_local = z_i.size(0)
        N_expected = self.batch_size * self.world_size
        if N_local != N_expected:
            raise ValueError(
                f"NT_Xent: got local batch {N_local}, expected {N_expected} "
                f"(batch_size={self.batch_size}, world_size={self.world_size})"
            )

        # Concatenate views: (2N, d)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix BEFORE temperature: (2N,2N) in [-1,1]
        sim_cos = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))

        # Compute β_t (either fixed or EVT-driven)
        beta_t = self._get_batch_beta(sim_cos)

        # -------- Base loss according to policy --------
        policy = self.policy.lower()
        if policy == "pl":
            logits_pl, labels_pl = self._build_pl_logits_and_labels(sim_cos)
            base_loss = self.ce(logits_pl, labels_pl)

        elif policy == "weibit":
            logits_wb, labels_wb = self._build_weibit_logits_and_labels(sim_cos, beta_t)
            base_loss = self.ce(logits_wb, labels_wb)

        elif policy == "soft":
            # build both heads and mix their losses
            logits_pl, labels_pl = self._build_pl_logits_and_labels(sim_cos)
            logits_wb, labels_wb = self._build_weibit_logits_and_labels(sim_cos, beta_t)
            loss_pl = self.ce(logits_pl, labels_pl)
            loss_wb = self.ce(logits_wb, labels_wb)
            lam = self.soft_lambda
            base_loss = (1.0 - lam) * loss_pl + lam * loss_wb

        elif policy == "gate":
            logits_gt, labels_gt = self._build_gate_logits_and_labels(sim_cos, beta_t)
            base_loss = self.ce(logits_gt, labels_gt)

        elif policy == "weibit_shrink":
            logits_ws, labels_ws = self._build_weibit_shrink_logits_and_labels(sim_cos, beta_t)
            base_loss = self.ce(logits_ws, labels_ws)

        elif policy in ("hard_select", "hard_selet", "hardselect"):
            # Binary per-anchor switch: PL(full negatives) vs Weibit(top-M)
            base_loss = self._hard_select_losses(sim_cos, beta_t)

        elif policy in ("soft_select", "softselect"):
            base_loss = self._soft_select_losses(sim_cos, beta_t, mode="both")

        elif policy in ("soft_select_rho", "softselect_rho"):
            base_loss = self._soft_select_losses(sim_cos, beta_t, mode="rho")

        elif policy in ("soft_select_aic", "softselect_aic"):
            base_loss = self._soft_select_losses(sim_cos, beta_t, mode="aic")

        elif policy in ("tlambda_select", "tlambda", "t_lambda_select", "t_lambda"):
            logits_tl, labels_tl = self._build_tlambda_logits_and_labels(sim_cos, mode="both")
            base_loss = self.ce(logits_tl, labels_tl)

        elif policy in ("tlambda_select_rho", "tlambda_rho", "t_lambda_rho"):
            logits_tl, labels_tl = self._build_tlambda_logits_and_labels(sim_cos, mode="rho")
            base_loss = self.ce(logits_tl, labels_tl)

        elif policy in ("tlambda_select_aic", "tlambda_aic", "t_lambda_aic"):
            logits_tl, labels_tl = self._build_tlambda_logits_and_labels(sim_cos, mode="aic")
            base_loss = self.ce(logits_tl, labels_tl)

        else:
            raise ValueError(f"Unknown NT_Xent policy: {self.policy}")

        loss = base_loss

        # -------- Optional Weibit top-M extreme regularizer --------
        if self.use_weib_topm and self.weib_lambda > 0.0:
            loss_weib_topm = self._weibull_topm_loss(sim_cos, beta_t)
            loss = loss + self.weib_lambda * loss_weib_topm
        
        # -------- Optional PL top-M extreme regularizer for ablation --------
        if self.use_pl_topm and self.pl_lambda > 0.0:
            loss_pl_topm = self._pl_topm_loss(sim_cos)
            loss = loss + self.pl_lambda * loss_pl_topm

        return loss