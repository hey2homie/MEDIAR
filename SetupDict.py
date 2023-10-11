import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import monai

import core
from train_tools import models

__all__ = ["TRAINER", "PREDICTOR", "MODELS", "OPTIMIZER", "SCHEDULER"]

TRAINER = {
    "mediar": core.MEDIAR.Trainer,
}

PREDICTOR = {
    "mediar": core.MEDIAR.Predictor,
    "ensemble_mediar": core.MEDIAR.EnsemblePredictor,
}

MODELS = {
    "unet": monai.networks.nets.UNet,
    "unetr": monai.networks.nets.unetr.UNETR,
    "swinunetr": monai.networks.nets.SwinUNETR,
    "mediar-former": models.MEDIARFormer,
}

OPTIMIZER = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}
