import torch
import torch.nn as nn
import torch.nn.functional as F


class _LambdaNet(nn.Module):
    """Small MLP that predicts a per-anchor gate lambda in (0,1)."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


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

    Additionally, an optional Weibull top-M regularizer can be added:

      L_total = L_policy + weib_lambda * L_weib_topM

    where L_weib_topM is an InfoNCE computed on a *restricted* choice set
    [positive | top-M hardest negatives] per anchor. This is the WEINCE-style
    "extreme" regularization term.

    The Weibit shape parameter beta is either:
      - fixed (weib_beta) if auto_weib_beta=False, or
      - estimated per batch from EVT diagnostics on the negative tail if
        auto_weib_beta=True (beta_t, with EMA and clamping).

    Top-M regularizer variants (added on top of the base loss):
      weib_topm_mode="weib"   : (default) use batch beta_t for all anchors.
      weib_topm_mode="gate"   : (4) anchor-wise gate lambda_i mixes PL-topM and
                                Weibit-topM per anchor.
      weib_topm_mode="shrink" : (5) anchor-wise beta_hat_i estimated from each
                                anchor's top-k tail, then shrunk towards beta_t.
    """

    def __init__(
        self,
        batch_size: int,
        temperature: float,
        world_size: int = 1,
        policy: str = "pl",             # "pl", "weibit", "soft"
        weib_beta: float = 4.0,         # initial / fixed β if auto_weib_beta=False
        soft_lambda: float = 0.5,       # mix weight for "soft" policy

        # --- WEINCE-style top-M regularizer (added on top of base loss) ---
        use_weib_topm: bool = False,
        weib_lambda: float = 0.0,       # weight for top-M regularizer
        weib_top_m: int = 16,           # number of hardest negatives per anchor

        eps: float = 1e-6,

        # --- PL top-M regularizer config (ablation) ---
        pl_lambda: float = 0.0,
        pl_top_m: int = 16,
        use_pl_topm: bool = False,

        # --- EVT / β diagnostics ---
        auto_weib_beta: bool = False,   # if True, estimate β_t from tail each batch
        beta_min: float = 0.5,
        beta_max: float = 16.0,
        tail_frac: float = 0.05,        # fraction of negative tail used for regression
        min_tail_points: int = 1024,    # minimum #tail points
        beta_ema_momentum: float = 0.9,
        r2_min: float = 0.85,           # minimum R^2 to trust a new β̂

        # --- (4) Anchor-wise gating for the top-M regularizer ---
        weib_topm_mode: str = "weib",   # {"weib","gate","shrink"}
        weib_gate_kappa: float = 10.0, # sigmoid steepness
        weib_gate_q: float = 0.2,      # quantile of rho used as pivot (smaller rho => harder)
        weib_gate_mix: str = "loss",   # {"loss","logit","hard_loss","hard_logit"}
        weib_gate_hard_threshold: float = 0.5,  # threshold for hard selection mixes

        # --- (5) Anchor-wise beta shrinkage for top-M regularizer ---
        weib_shrink_k: int = 64,        # top-k negatives used to estimate anchor beta_hat_i
        weib_shrink_alpha: float = 0.5, # shrink weight toward beta_hat_i (rest toward beta_t)
        weib_shrink_r2_min: float = 0.2,# trust threshold for anchor-wise fit

        # --- New: lambda_i estimation variants for gate modes ---
        weib_gate_estimator: str = "rho",   # {"rho","rho_knn","mlp","mlp_knn","aic_knn","mlp_distill_aic_knn"}

        # kNN smoothing for lambda_i (works for *_knn estimators)
        weib_gate_knn_k: int = 8,
        weib_gate_knn_eta: float = 0.5,
        weib_gate_knn_temp: float = 0.1,

        # MLP lambda head (works for "mlp" / "mlp_knn")
        weib_gate_mlp_hidden: int = 128,
        weib_gate_mlp_detach: bool = True,
        weib_gate_mlp_lambda_max: float = 1.0,
        gate_embed_dim: int = 0,

        # Regularizers for learned lambda
        weib_gate_smooth_weight: float = 0.0,
        weib_gate_target_mean: float = -1.0,
        weib_gate_mean_weight: float = 0.0,
        weib_gate_entropy_weight: float = 0.0,

        # Teacher student distillation (AIC teacher computed from kNN pooled tails)
        weib_gate_distill_weight: float = 0.0,
        weib_gate_aic_tail_k: int = 0,
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

        # Top-M mode config
        self.weib_topm_mode = str(weib_topm_mode).lower()
        self.weib_gate_kappa = float(weib_gate_kappa)
        self.weib_gate_q = float(weib_gate_q)
        self.weib_gate_mix = str(weib_gate_mix).lower()
        self.weib_gate_hard_threshold = float(weib_gate_hard_threshold)

        self.weib_shrink_k = int(weib_shrink_k)
        self.weib_shrink_alpha = float(weib_shrink_alpha)
        self.weib_shrink_r2_min = float(weib_shrink_r2_min)

        # Lambda estimation config
        self.weib_gate_estimator = str(weib_gate_estimator).lower()
        self.weib_gate_knn_k = int(weib_gate_knn_k)
        self.weib_gate_knn_eta = float(weib_gate_knn_eta)
        self.weib_gate_knn_temp = float(weib_gate_knn_temp)

        self.weib_gate_mlp_hidden = int(weib_gate_mlp_hidden)
        self.weib_gate_mlp_detach = bool(weib_gate_mlp_detach)
        self.weib_gate_mlp_lambda_max = float(weib_gate_mlp_lambda_max)

        self.weib_gate_smooth_weight = float(weib_gate_smooth_weight)
        self.weib_gate_target_mean = float(weib_gate_target_mean)
        self.weib_gate_mean_weight = float(weib_gate_mean_weight)
        self.weib_gate_entropy_weight = float(weib_gate_entropy_weight)

        self.weib_gate_distill_weight = float(weib_gate_distill_weight)
        self.weib_gate_aic_tail_k = int(weib_gate_aic_tail_k)

        # If we will use an MLP gate, create it if we know the embedding dim.
        # gate_embed_dim should usually be projection_dim (z dim).
        self.lambda_net = None
        self._lambda_in_dim = None
        self._lambda_stats_dim = 8  # keep in sync with _build_gate_stats
        if self.weib_gate_estimator in {"mlp", "mlp_knn", "mlp_distill_aic_knn"}:
            if gate_embed_dim and int(gate_embed_dim) > 0:
                self._init_lambda_net(int(gate_embed_dim))

        # Running β_t (EMA) and last diagnostics
        self.register_buffer("beta_ema", torch.tensor(float(weib_beta)))
        self.last_beta_hat = None
        self.last_beta_t = float(weib_beta)
        self.last_R2 = None
        self.last_rho = None

        # Top-M diagnostics (optional, useful for logging)
        self.last_gate_lambda_mean = None
        self.last_beta_hat_anchor_mean = None
        self.last_beta_hat_anchor_r2_mean = None
        self.last_beta_eff_anchor_mean = None

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

    def _build_pl_logits_and_labels(self, sim_cos: torch.Tensor):
        """Build PL (standard NT-Xent) logits and labels."""
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

    def _build_weibit_logits_and_labels(self, sim_cos: torch.Tensor, beta_t: float):
        """Build Weibit logits and labels using batch-dependent beta_t."""
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

    # ----------------------- Top-M regularizer helpers -----------------------

    @torch.no_grad()
    def _anchorwise_gate_lambda(self, rho: torch.Tensor) -> torch.Tensor:
        """
        (4) Compute an anchor-wise gate lambda_i in [0,1] from cap proximity rho_i.

        rho_i is a shortfall proxy: rho_i = 1 - max_j s_{i,j} (smaller => closer to cap).

        We set a pivot rho0 as the q-quantile of rho across anchors, then:
            lambda_i = sigmoid(kappa * (rho0 - rho_i))

        So anchors with *harder* negatives (smaller rho) get larger lambda.
        Returned tensor has shape (2N,).
        """
        rho_det = rho.detach().float()
        # robust quantile via sort (avoid torch.quantile dependency)
        rho_sorted, _ = torch.sort(rho_det)
        q = float(self.weib_gate_q)
        q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
        k = int(round(q * (rho_sorted.numel() - 1)))
        rho0 = rho_sorted[k]
        lam = torch.sigmoid(self.weib_gate_kappa * (rho0 - rho_det))
        return lam

    def _init_lambda_net(self, embed_dim: int):
        if self.lambda_net is not None:
            return
        in_dim = int(embed_dim) + int(self._lambda_stats_dim)
        self.lambda_net = _LambdaNet(in_dim=in_dim, hidden=self.weib_gate_mlp_hidden)
        self._lambda_in_dim = in_dim

    def _build_gate_stats(
        self,
        s_pos: torch.Tensor,
        top_vals: torch.Tensor,
        rho: torch.Tensor,
        beta_t: float,
    ) -> torch.Tensor:
        # s_pos: (2N,1), top_vals: (2N,M), rho: (2N,)
        s_pos_ = s_pos.squeeze(1)
        s_max = top_vals[:, 0]
        if top_vals.size(1) > 1:
            s_2nd = top_vals[:, 1]
        else:
            s_2nd = s_max
        gap = s_pos_ - s_max
        top_mean = top_vals.mean(dim=1)
        top_std = top_vals.std(dim=1, unbiased=False)
        log_rho = torch.log(rho.clamp_min(self.eps))
        tau = torch.full_like(s_max, float(self.temperature))
        beta = torch.full_like(s_max, float(beta_t))

        # Keep D=8
        stats = torch.stack(
            [s_pos_, s_max, s_2nd, gap, top_mean, top_std, log_rho, tau],
            dim=1,
        )
        return stats

    def _knn_indices_and_weights(self, z: torch.Tensor):
        B = z.size(0)
        k = int(self.weib_gate_knn_k)
        if k <= 0:
            return None, None
        k = min(k, max(1, B - 1))

        z_det = z.detach()
        z_det = F.normalize(z_det, dim=1)
        sim = z_det @ z_det.t()
        sim.fill_diagonal_(-1e9)

        vals, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        temp = max(float(self.weib_gate_knn_temp), 1e-6)
        w = torch.exp(vals / temp)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return idx, w

    def _apply_knn_smoothing(self, lam: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eta = float(self.weib_gate_knn_eta)
        if eta <= 0.0:
            return lam
        idx, w = self._knn_indices_and_weights(z)
        if idx is None:
            return lam
        lam_n = lam[idx]
        neigh = (w * lam_n).sum(dim=1)
        return (1.0 - eta) * lam + eta * neigh

    def _knn_smoothness_penalty(self, lam: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        idx, w = self._knn_indices_and_weights(z)
        if idx is None:
            return lam.new_tensor(0.0)
        lam_n = lam[idx]
        diff2 = (lam.unsqueeze(1) - lam_n) ** 2
        return (w * diff2).mean()

    def _gate_prior_penalty(self, lam: torch.Tensor) -> torch.Tensor:
        reg = lam.new_tensor(0.0)
        if self.weib_gate_mean_weight > 0.0 and self.weib_gate_target_mean >= 0.0:
            tgt = float(self.weib_gate_target_mean)
            reg = reg + float(self.weib_gate_mean_weight) * (lam.mean() - tgt) ** 2
        if self.weib_gate_entropy_weight != 0.0:
            w = float(self.weib_gate_entropy_weight)
            lam_c = lam.clamp(1e-6, 1.0 - 1e-6)
            entropy = -(lam_c * torch.log(lam_c) + (1.0 - lam_c) * torch.log(1.0 - lam_c)).mean()
            reg = reg - w * entropy
        return reg

    @torch.no_grad()
    def _aic_teacher_lambda_from_pooled_tail(
        self,
        z: torch.Tensor,
        top_vals_tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a teacher lambda_i using kNN pooled tail diagnostics.

        For each anchor i, we pool the top-k tail similarities from i and its kNN
        neighbors in embedding space, fit both:
          - Weibull: log Fhat ~ a + beta log(delta)
          - Gumbel:  log(-log Fhat) ~ a + slope log(delta)
        and convert the AIC difference into an Akaike weight for Weibull.

        Returns:
          lam_teacher: (B,) in [0,1]
        """
        B = top_vals_tail.size(0)
        k_tail = top_vals_tail.size(1)

        idx, _ = self._knn_indices_and_weights(z)
        if idx is None:
            # fall back to no pooling
            idx = torch.empty((B, 0), dtype=torch.long, device=z.device)

        # include self
        self_idx = torch.arange(B, device=z.device).unsqueeze(1)
        idx_all = torch.cat([self_idx, idx], dim=1)  # (B, 1+K)

        pooled = top_vals_tail[idx_all]  # (B, 1+K, k_tail)
        delta = (1.0 - pooled).clamp_min(self.eps)
        delta = delta.reshape(B, -1)

        # sort ascending deltas so empirical CDF aligns with order statistics
        delta_sorted, _ = torch.sort(delta, dim=1)
        L = delta_sorted.size(1)
        if L < 10:
            return torch.full((B,), 0.0, device=z.device, dtype=delta_sorted.dtype)

        x = torch.log(delta_sorted)

        k = torch.arange(1, L + 1, device=z.device, dtype=x.dtype)
        Fhat = k / (L + 1.0)
        yW = torch.log(Fhat)
        yG = torch.log((-torch.log(Fhat)).clamp_min(1e-12))

        burn = max(1, L // 20)
        xb = x[:, burn:]
        yWb = yW[burn:]
        yGb = yG[burn:]

        # Weibull fit
        yW_mean = yWb.mean()
        x_mean = xb.mean(dim=1, keepdim=True)
        xc = xb - x_mean
        yWc = yWb - yW_mean
        den = (xc * xc).sum(dim=1).clamp_min(1e-12)
        numW = (xc * yWc.unsqueeze(0)).sum(dim=1)
        slopeW = numW / den
        yW_hat = slopeW.unsqueeze(1) * xc + yW_mean
        rssW = ((yWb.unsqueeze(0) - yW_hat) ** 2).sum(dim=1).clamp_min(1e-12)

        # Gumbel fit
        yG_mean = yGb.mean()
        yGc = yGb - yG_mean
        numG = (xc * yGc.unsqueeze(0)).sum(dim=1)
        slopeG = numG / den
        yG_hat = slopeG.unsqueeze(1) * xc + yG_mean
        rssG = ((yGb.unsqueeze(0) - yG_hat) ** 2).sum(dim=1).clamp_min(1e-12)

        # AIC difference: AIC_G - AIC_W = n log(RSS_G/n) - n log(RSS_W/n)
        n = float(yWb.numel())
        dAIC = n * (torch.log(rssG / n) - torch.log(rssW / n))

        # Akaike weight for Weibull
        lam_teacher = torch.sigmoid(0.5 * dAIC)
        return lam_teacher

    def _compute_anchor_lambda(
        self,
        z: torch.Tensor,
        s_pos: torch.Tensor,
        top_vals: torch.Tensor,
        top_vals_tail: torch.Tensor,
        rho: torch.Tensor,
        beta_t: float,
    ):
        est = self.weib_gate_estimator
        gate_reg = z.new_tensor(0.0)

        if est in {"rho", "rho_knn"}:
            with torch.no_grad():
                lam = self._anchorwise_gate_lambda(rho)
            if est == "rho_knn":
                lam = self._apply_knn_smoothing(lam, z)
            with torch.no_grad():
                self.last_gate_lambda_mean = float(lam.mean().item())
            return lam, gate_reg

        if est in {"aic_knn"}:
            with torch.no_grad():
                # Use pooled tails for a stable small-sample teacher
                lam = self._aic_teacher_lambda_from_pooled_tail(z, top_vals_tail)
            # Optional extra smoothing (controlled by weib_gate_knn_eta)
            lam = self._apply_knn_smoothing(lam, z)
            with torch.no_grad():
                self.last_gate_lambda_mean = float(lam.mean().item())
            return lam, gate_reg

        if est in {"mlp", "mlp_knn", "mlp_distill_aic_knn"}:
            if self.lambda_net is None:
                self._init_lambda_net(embed_dim=int(z.size(1)))
            if self.lambda_net is None:
                raise RuntimeError("lambda_net failed to initialize")

            # Ensure the gate is on the correct device
            if next(self.lambda_net.parameters()).device != z.device:
                self.lambda_net = self.lambda_net.to(device=z.device)

            z_in = z.detach() if self.weib_gate_mlp_detach else z
            stats = self._build_gate_stats(s_pos, top_vals, rho, beta_t)
            stats = stats.detach() if self.weib_gate_mlp_detach else stats

            x = torch.cat([z_in, stats], dim=1)
            lam01 = self.lambda_net(x)
            lam = float(self.weib_gate_mlp_lambda_max) * lam01
            lam = lam.clamp(0.0, 1.0)

            # Optional distillation against AIC teacher
            if est == "mlp_distill_aic_knn" and self.weib_gate_distill_weight > 0.0:
                with torch.no_grad():
                    lam_teacher = self._aic_teacher_lambda_from_pooled_tail(z, top_vals_tail)
                gate_reg = gate_reg + float(self.weib_gate_distill_weight) * F.mse_loss(lam, lam_teacher, reduction="mean")

            if est in {"mlp_knn", "mlp_distill_aic_knn"}:
                lam = self._apply_knn_smoothing(lam, z)

            if self.weib_gate_smooth_weight > 0.0:
                gate_reg = gate_reg + float(self.weib_gate_smooth_weight) * self._knn_smoothness_penalty(lam, z)
            gate_reg = gate_reg + self._gate_prior_penalty(lam)

            with torch.no_grad():
                self.last_gate_lambda_mean = float(lam.mean().item())

            return lam, gate_reg

        raise ValueError(f"Unknown weib_gate_estimator: {self.weib_gate_estimator}")

    @torch.no_grad()
    def _estimate_anchor_beta_from_topk(self, top_vals_tail: torch.Tensor):
        """
        (5) Estimate anchor-wise beta_hat_i from each row's top-k negative similarities.

        top_vals_tail: (2N, k_tail), sorted descending similarities for each anchor.
        We convert to shortfalls delta = 1 - s and fit:
            log F(delta) ≈ log c + beta * log delta
        using a simple least squares slope on (log delta, log Fhat).

        Returns:
            beta_hat: (2N,) nonnegative
            R2:       (2N,) in [0,1]
        """
        device = top_vals_tail.device
        eps = self.eps

        # delta is small for large similarities
        delta = (1.0 - top_vals_tail.detach()).clamp_min(eps)  # (2N,k)
        k_tail = delta.size(1)
        if k_tail < 10:
            beta_hat = torch.full((delta.size(0),), float(self.beta_ema.item()), device=device)
            R2 = torch.zeros_like(beta_hat)
            return beta_hat, R2

        # already approximately sorted (since top_vals_tail is sorted desc)
        x = torch.log(delta)  # (2N,k)
        k = torch.arange(1, k_tail + 1, device=device, dtype=x.dtype)  # 1..k
        Fhat = k / (k_tail + 1.0)
        y = torch.log(Fhat)  # (k,)

        burn = max(1, k_tail // 20)
        xb = x[:, burn:]
        yb = y[burn:]  # (L,)
        L = yb.numel()
        if L < 2:
            beta_hat = torch.full((delta.size(0),), float(self.beta_ema.item()), device=device)
            R2 = torch.zeros_like(beta_hat)
            return beta_hat, R2

        ymean = yb.mean()
        xmean = xb.mean(dim=1, keepdim=True)

        xc = xb - xmean
        yc = yb - ymean  # (L,)

        den = (xc * xc).sum(dim=1).clamp_min(1e-12)     # (2N,)
        num = (xc * yc.unsqueeze(0)).sum(dim=1)         # (2N,)
        beta_hat = (num / den).clamp_min(0.0)           # (2N,)

        # R^2 per anchor
        slope = beta_hat.unsqueeze(1)                   # (2N,1)
        yhat = slope * xc + ymean                       # (2N,L)
        ss_res = ((yb.unsqueeze(0) - yhat) ** 2).sum(dim=1)  # (2N,)
        ss_tot = ((yb - ymean) ** 2).sum().clamp_min(1e-12)  # scalar
        R2 = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)    # (2N,)

        return beta_hat, R2


    def _build_weibit_shrink_logits_and_labels(self, sim_cos: torch.Tensor, beta_t: float):
        """
        (5) Build logits/labels for the *top-M* Weibit regularizer using anchor-wise beta shrinkage.

        This is a helper to make the shrink-mode implementation explicit and reusable.

        Returns:
            logits: (2N, 1+M) over [positive | top-M negatives]
            labels: (2N,) all zeros (positive is index 0)
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # negatives mask
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos.masked_fill(~mask, -1e9)
        num_valid = N2 - 2

        top_m = min(int(self.weib_top_m), num_valid)
        top_m = max(1, top_m)

        k_tail = max(int(self.weib_shrink_k), top_m)
        k_tail = min(k_tail, num_valid)
        k_tail = max(10, k_tail)

        # top-k tail (sorted desc); reuse first M cols as top-M negatives
        top_vals_tail, _ = torch.topk(sim_neg, k=k_tail, dim=1, largest=True, sorted=True)  # (2N,k_tail)
        top_vals = top_vals_tail[:, :top_m]  # (2N,M)
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        # Estimate anchor-wise betas from tail (no grad)
        beta_hat, R2 = self._estimate_anchor_beta_from_topk(top_vals_tail)

        # trust weight from R2
        if self.weib_shrink_r2_min >= 1.0:
            trust = torch.zeros_like(R2)
        else:
            trust = ((R2 - self.weib_shrink_r2_min) / (1.0 - self.weib_shrink_r2_min)).clamp(0.0, 1.0)

        alpha = float(self.weib_shrink_alpha)
        alpha_i = alpha * trust  # (2N,)

        beta_t_val = float(beta_t)
        beta_eff = (1.0 - alpha_i) * beta_t_val + alpha_i * beta_hat
        beta_eff = beta_eff.clamp(self.beta_min, self.beta_max)

        # Store diagnostics
        with torch.no_grad():
            self.last_beta_hat_anchor_mean = float(beta_hat.mean().item())
            self.last_beta_hat_anchor_r2_mean = float(R2.mean().item())
            self.last_beta_eff_anchor_mean = float(beta_eff.mean().item())

        # Weibit logits with anchor-wise beta_eff
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        s_neg_c = top_vals.clamp(max=1.0 - eps)

        beta_col = beta_eff.to(device=device, dtype=s_pos_c.dtype).unsqueeze(1)  # (2N,1)
        pos_logits = -beta_col * torch.log(1.0 - s_pos_c)    # (2N,1)
        neg_logits = -beta_col * torch.log(1.0 - s_neg_c)    # (2N,M)
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (2N,1+M)

        labels = torch.zeros(N2, dtype=torch.long, device=device)
        return logits, labels
    
    def _weibull_topm_loss(self, z: torch.Tensor, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
            """
            Top-M regularizer. Dispatches based on weib_topm_mode:

              - "weib":       Weibit top-M with batch beta_t (original).
              - "gate":       anchor-wise gated mix of PL-topM and Weibit-topM (as before).
              - "gate_weib":  (improved) anchor-wise *weighting* of Weibit-topM only (no PL inside reg).
              - "shrink":     anchor-wise beta_hat_i (from top-k tail) shrunk towards beta_t (as before).
              - "shrink_gate":(improved) shrink + gate: anchor-wise beta_eff AND anchor-wise weighting.

            NOTE: The base loss is still controlled by `policy` ("pl"/"weibit"/"soft").
                  This only affects the additional top-M regularization term when enabled.
            """
            mode = self.weib_topm_mode
            if mode == "weib":
                return self._weibull_topm_loss_batch(sim_cos, beta_t)
            if mode == "gate":
                return self._weibull_topm_loss_gate(z, sim_cos, beta_t)
            if mode == "gate_weib":
                return self._weibull_topm_loss_gate_weib(z, sim_cos, beta_t)
            if mode == "shrink":
                return self._weibull_topm_loss_shrink(sim_cos, beta_t)
            if mode == "shrink_gate":
                return self._weibull_topm_loss_shrink_gate(z, sim_cos, beta_t)
            raise ValueError(f"Unknown weib_topm_mode: {self.weib_topm_mode}")

    def _weibull_topm_loss_batch(self, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
        """Original Weibit top-M loss on cosine similarities, using batch beta_t."""
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
        num_valid = N2 - 2
        top_m = min(int(self.weib_top_m), num_valid)
        top_m = max(1, top_m)

        # top-M hardest negatives
        top_vals, _ = torch.topk(sim_neg, k=top_m, dim=1, largest=True, sorted=True)  # (2N,M)

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

    def _weibull_topm_loss_gate(self, z: torch.Tensor, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
        """
        (4) Anchor-wise gate for the top-M regularizer.

        We compute a per-anchor lambda_i using rho_i = 1 - max_neg_similarity_i, then
        mix either:
          - losses: (1-lam_i)*CE(PL_topM) + lam_i*CE(Weibit_topM), or
          - logits: logits = (1-lam_i)*pl_logits + lam_i*weib_logits.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # negatives mask
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos.masked_fill(~mask, -1e9)
        num_valid = N2 - 2
        top_m = min(int(self.weib_top_m), num_valid)
        top_m = max(1, top_m)

        # top-M negatives (sorted desc); for AIC teacher we may need a longer tail
        need_tail = self.weib_gate_estimator in {"aic_knn", "mlp_distill_aic_knn"}
        if need_tail:
            k_tail = int(self.weib_gate_aic_tail_k) if int(self.weib_gate_aic_tail_k) > 0 else int(self.weib_shrink_k)
            k_tail = max(k_tail, top_m)
            k_tail = min(k_tail, num_valid)
            k_tail = max(10, k_tail)
            top_vals_tail, _ = torch.topk(sim_neg, k=k_tail, dim=1, largest=True, sorted=True)  # (2N,k_tail)
            top_vals = top_vals_tail[:, :top_m]
        else:
            top_vals, _ = torch.topk(sim_neg, k=top_m, dim=1, largest=True, sorted=True)  # (2N,M)
            top_vals_tail = top_vals
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        # rho_i = 1 - max_neg
        s_max_neg = top_vals.detach()[:, 0].clamp(max=1.0)  # (2N,)
        rho = (1.0 - s_max_neg).clamp_min(self.eps)         # (2N,)
        lam, gate_reg = self._compute_anchor_lambda(
            z, s_pos, top_vals.detach(), top_vals_tail.detach(), rho, beta_t
        )

        labels = torch.zeros(N2, dtype=torch.long, device=device)

        # Build PL logits over [pos | topM negs]
        pl_logits = torch.cat([s_pos / self.temperature, top_vals / self.temperature], dim=1)

        # Build Weibit logits over [pos | topM negs]
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        top_vals_c = top_vals.clamp(max=1.0 - eps)
        beta = float(beta_t)
        weib_logits = torch.cat(
            [-beta * torch.log(1.0 - s_pos_c), -beta * torch.log(1.0 - top_vals_c)],
            dim=1,
        )

        mix = self.weib_gate_mix
        if mix == "logit":
            lam_col = lam.to(device=device, dtype=pl_logits.dtype).unsqueeze(1)  # (2N,1)
            logits = (1.0 - lam_col) * pl_logits + lam_col * weib_logits
            loss = F.cross_entropy(logits, labels, reduction="mean")
            return loss + gate_reg

        if mix == "loss":
            loss_pl = F.cross_entropy(pl_logits, labels, reduction="none")    # (2N,)
            loss_wb = F.cross_entropy(weib_logits, labels, reduction="none")  # (2N,)
            lam_f = lam.to(device=device, dtype=loss_pl.dtype)
            loss = ((1.0 - lam_f) * loss_pl + lam_f * loss_wb).mean()
            return loss + gate_reg

        # Hard model selection: choose exactly one expert per anchor.
        # We use a straight-through (ST) gate so gradients still flow into lambda.
        if mix in {"hard_loss", "hard_logit"}:
            thr = float(self.weib_gate_hard_threshold)
            lam_hard = (lam > thr).to(device=device, dtype=pl_logits.dtype)  # (2N,)
            # Straight-through: forward uses hard gate, backward uses soft lam.
            lam_st = lam_hard + (lam.to(device=device, dtype=pl_logits.dtype) - lam.to(device=device, dtype=pl_logits.dtype).detach())

            if mix == "hard_logit":
                lam_col = lam_st.unsqueeze(1)
                logits = (1.0 - lam_col) * pl_logits + lam_col * weib_logits
                loss = F.cross_entropy(logits, labels, reduction="mean")
                return loss + gate_reg

            # hard_loss
            loss_pl = F.cross_entropy(pl_logits, labels, reduction="none")    # (2N,)
            loss_wb = F.cross_entropy(weib_logits, labels, reduction="none")  # (2N,)
            lam_st_f = lam_st.to(device=device, dtype=loss_pl.dtype)
            loss = ((1.0 - lam_st_f) * loss_pl + lam_st_f * loss_wb).mean()
            return loss + gate_reg

        raise ValueError(f"Unknown weib_gate_mix: {self.weib_gate_mix}")

    def _weibull_topm_loss_gate_weib(self, z: torch.Tensor, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
        """
        (Improved) Anchor-wise gate for the top-M regularizer, but **Weibit-only**.

        Motivation: The base loss (usually PL/InfoNCE) is already present, so mixing PL-topM
        inside the regularizer can dilute the extreme-value correction. Here we instead:
            - compute an anchor-wise gate lambda_i from rho_i = 1 - max_neg_similarity_i
            - compute per-anchor Weibit-topM cross-entropy
            - weight per-anchor losses by lambda_i (normalized)

        This tends to behave like an "adaptive" regularizer that turns on for anchors whose
        hardest negatives are closest to the cosine cap.
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # negatives mask
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos.masked_fill(~mask, -1e9)
        num_valid = N2 - 2
        top_m = min(int(self.weib_top_m), num_valid)
        top_m = max(1, top_m)

        # top-M negatives (sorted desc); for AIC teacher we may need a longer tail
        need_tail = self.weib_gate_estimator in {"aic_knn", "mlp_distill_aic_knn"}
        if need_tail:
            k_tail = int(self.weib_gate_aic_tail_k) if int(self.weib_gate_aic_tail_k) > 0 else int(self.weib_shrink_k)
            k_tail = max(k_tail, top_m)
            k_tail = min(k_tail, num_valid)
            k_tail = max(10, k_tail)
            top_vals_tail, _ = torch.topk(sim_neg, k=k_tail, dim=1, largest=True, sorted=True)  # (2N,k_tail)
            top_vals = top_vals_tail[:, :top_m]
        else:
            top_vals, _ = torch.topk(sim_neg, k=top_m, dim=1, largest=True, sorted=True)  # (2N,M)
            top_vals_tail = top_vals
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        # rho_i = 1 - max_neg
        s_max_neg = top_vals.detach()[:, 0].clamp(max=1.0)  # (2N,)
        rho = (1.0 - s_max_neg).clamp_min(self.eps)         # (2N,)
        lam, gate_reg = self._compute_anchor_lambda(
            z, s_pos, top_vals.detach(), top_vals_tail.detach(), rho, beta_t
        )

        # Weibit logits over [pos | topM negs]
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        top_vals_c = top_vals.clamp(max=1.0 - eps)
        beta = float(beta_t)
        logits = torch.cat(
            [-beta * torch.log(1.0 - s_pos_c), -beta * torch.log(1.0 - top_vals_c)],
            dim=1,
        )  # (2N,1+M)

        labels = torch.zeros(N2, dtype=torch.long, device=device)

        # Per-anchor CE, weighted + normalized
        per = F.cross_entropy(logits, labels, reduction="none")  # (2N,)
        lam_f = lam.to(device=device, dtype=per.dtype)
        loss = (lam_f * per).sum() / (lam_f.sum() + 1e-12)
        return loss + gate_reg


    def _weibull_topm_loss_shrink_gate(self, z: torch.Tensor, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
        """
        (Improved) Combine **shrink** and **gate** inside the top-M regularizer.

        - shrink: estimate anchor-wise beta_hat_i from each anchor's negative tail and
          shrink toward batch beta_t to reduce variance / instability.
        - gate:   weight the regularizer per anchor according to rho_i = 1 - max_neg_similarity_i.

        Concretely:
            beta_eff_i = (1-alpha_i) * beta_t + alpha_i * beta_hat_i
            L = sum_i lambda_i * CE_i(beta_eff_i) / sum_i lambda_i
        """
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        # negatives mask
        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos.masked_fill(~mask, -1e9)
        num_valid = N2 - 2

        top_m = min(int(self.weib_top_m), num_valid)
        top_m = max(1, top_m)

        k_tail = max(int(self.weib_shrink_k), top_m)
        k_tail = min(k_tail, num_valid)
        k_tail = max(10, k_tail)

        # top-k tail (sorted desc); reuse first M cols as top-M negatives
        top_vals_tail, _ = torch.topk(sim_neg, k=k_tail, dim=1, largest=True, sorted=True)  # (2N,k_tail)
        top_vals = top_vals_tail[:, :top_m]  # (2N,M)
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        # Gate from rho_i = 1 - max_neg
        s_max_neg = top_vals.detach()[:, 0].clamp(max=1.0)
        rho = (1.0 - s_max_neg).clamp_min(self.eps)
        lam, gate_reg = self._compute_anchor_lambda(
            z, s_pos, top_vals.detach(), top_vals_tail.detach(), rho, beta_t
        )

        # Estimate anchor-wise betas from tail (no grad)
        beta_hat, R2 = self._estimate_anchor_beta_from_topk(top_vals_tail)

        # trust weight from R2 (same as shrink mode)
        if self.weib_shrink_r2_min >= 1.0:
            trust = torch.zeros_like(R2)
        else:
            trust = ((R2 - self.weib_shrink_r2_min) / (1.0 - self.weib_shrink_r2_min)).clamp(0.0, 1.0)

        alpha = float(self.weib_shrink_alpha)
        alpha_i = alpha * trust  # (2N,)

        beta_t_val = float(beta_t)
        beta_eff = (1.0 - alpha_i) * beta_t_val + alpha_i * beta_hat
        beta_eff = beta_eff.clamp(self.beta_min, self.beta_max)

        # Store diagnostics
        with torch.no_grad():
            self.last_beta_hat_anchor_mean = float(beta_hat.mean().item())
            self.last_beta_hat_anchor_r2_mean = float(R2.mean().item())
            self.last_beta_eff_anchor_mean = float(beta_eff.mean().item())

        # Weibit logits with anchor-wise beta_eff
        eps = self.eps
        s_pos_c = s_pos.clamp(max=1.0 - eps)
        s_neg_c = top_vals.clamp(max=1.0 - eps)

        beta_col = beta_eff.to(device=device, dtype=s_pos_c.dtype).unsqueeze(1)  # (2N,1)
        pos_logits = -beta_col * torch.log(1.0 - s_pos_c)    # (2N,1)
        neg_logits = -beta_col * torch.log(1.0 - s_neg_c)    # (2N,M)
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (2N,1+M)

        labels = torch.zeros(N2, dtype=torch.long, device=device)
        per = F.cross_entropy(logits, labels, reduction="none")  # (2N,)
        lam_f = lam.to(device=device, dtype=per.dtype)
        loss = (lam_f * per).sum() / (lam_f.sum() + 1e-12)
        return loss + gate_reg

    def _weibull_topm_loss_shrink(self, sim_cos: torch.Tensor, beta_t: float) -> torch.Tensor:
        """
        (5) Anchor-wise beta shrinkage for the top-M regularizer.

        This calls :meth:`_build_weibit_shrink_logits_and_labels` to build the logits/labels,
        then applies a cross-entropy loss.
        """
        logits, labels = self._build_weibit_shrink_logits_and_labels(sim_cos, beta_t)
        return F.cross_entropy(logits, labels, reduction="mean")

    def _pl_topm_loss(self, sim_cos: torch.Tensor) -> torch.Tensor:
        """PL top-M loss (ablation): softmax on [pos | top-M negs]."""
        device = sim_cos.device
        N2 = sim_cos.size(0)
        N = N2 // 2

        idx = torch.arange(N2, device=device)
        pos_idx = (idx + N) % N2

        mask = torch.ones_like(sim_cos, dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos_idx] = False

        sim_neg = sim_cos.masked_fill(~mask, -1e9)
        num_valid = N2 - 2
        top_m = min(int(self.pl_top_m), num_valid)
        top_m = max(1, top_m)

        top_vals, _ = torch.topk(sim_neg, k=top_m, dim=1, largest=True, sorted=True)  # (2N,M)
        s_pos = sim_cos[idx, pos_idx].unsqueeze(1)  # (2N,1)

        logits = torch.cat([s_pos / self.temperature, top_vals / self.temperature], dim=1)  # (2N,1+M)
        labels = torch.zeros(N2, dtype=torch.long, device=device)
        return self.ce(logits, labels)

    # ------------------------------- forward -------------------------------

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
            logits_pl, labels_pl = self._build_pl_logits_and_labels(sim_cos)
            logits_wb, labels_wb = self._build_weibit_logits_and_labels(sim_cos, beta_t)
            loss_pl = self.ce(logits_pl, labels_pl)
            loss_wb = self.ce(logits_wb, labels_wb)
            lam = self.soft_lambda
            base_loss = (1.0 - lam) * loss_pl + lam * loss_wb

        else:
            raise ValueError(f"Unknown NT_Xent policy: {self.policy}")

        loss = base_loss

        # -------- Optional Weibit top-M extreme regularizer --------
        if self.use_weib_topm and self.weib_lambda > 0.0:
            loss_weib_topm = self._weibull_topm_loss(z, sim_cos, beta_t)
            loss = loss + self.weib_lambda * loss_weib_topm

        # -------- Optional PL top-M extreme regularizer for ablation --------
        if self.use_pl_topm and self.pl_lambda > 0.0:
            loss_pl_topm = self._pl_topm_loss(sim_cos)
            loss = loss + self.pl_lambda * loss_pl_topm

        return loss