"""Modèles de simulation de croissance tumorale."""

from simulation.base import TumorGrowthModel
from simulation.calibration import (
    CalibrationResult,
    FisherKPPCalibrationResult,
    calibrate_fisher_kpp,
    calibrate_gompertz,
    generate_noisy_fisher_kpp_observations,
    generate_noisy_gompertz_observations,
)
from simulation.diffusion_reaction import (
    FisherKPPModel,
    FisherKPPResult,
    simulate_2d,
)
from simulation.gompertz import GompertzModel, GompertzResult, simulate
from simulation.surrogate import (
    SurrogateDataset,
    TrainedSurrogate,
    generate_fisher_kpp_surrogate_dataset,
    generate_gompertz_surrogate_dataset,
    train_surrogate,
)

# Alias de compatibilité : le modèle de diffusion-réaction est implémenté par
# FisherKPPModel (équation de Fisher-Kolmogorov-Petrovsky-Piskunov). Les deux
# noms désignent la même classe.
DiffusionReactionModel = FisherKPPModel

__all__ = [
    "TumorGrowthModel",
    "GompertzModel",
    "GompertzResult",
    "simulate",
    "FisherKPPModel",
    "DiffusionReactionModel",
    "FisherKPPResult",
    "simulate_2d",
    "CalibrationResult",
    "calibrate_gompertz",
    "generate_noisy_gompertz_observations",
    "FisherKPPCalibrationResult",
    "calibrate_fisher_kpp",
    "generate_noisy_fisher_kpp_observations",
    "SurrogateDataset",
    "TrainedSurrogate",
    "generate_gompertz_surrogate_dataset",
    "generate_fisher_kpp_surrogate_dataset",
    "train_surrogate",
]