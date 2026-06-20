import torchvision
import torch.nn as nn



def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


def get_resnet_adaptable(
    name: str,
    dataset: str,
    pretrained: bool = False,
) -> nn.Module:
    """
    Return a ResNet 18 or 50 with an appropriate stem for the dataset.

    dataset:
      "CIFAR10", "CIFAR100", "ImageNet32"  -> CIFAR style stem
      "STL10", "ImageNet"                  -> standard ImageNet stem
    """

    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
    else:
        raise KeyError(f"{name} is not a valid ResNet version")

    dataset = dataset.lower()

    small_image_datasets = {"cifar10", "cifar100", "imagenet32"}

    if dataset in small_image_datasets:
        # CIFAR style stem: better for 32 x 32 inputs
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

    # STL10 and ImageNet keep the default 7 x 7 stride 2 conv and maxpool

    return model

