"""Interface commune aux modèles de croissance tumorale du projet."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class TumorGrowthModel(ABC):
    """Interface commune minimale à tous les modèles de croissance tumorale.

    Un modèle concret valide et stocke ses paramètres dans son constructeur,
    puis expose une méthode `simulate()` sans argument qui relance la
    simulation et retourne un résultat structuré propre au modèle.

    Cette interface reste volontairement légère : elle sert de point
    d'extension pour ajouter facilement de futurs modèles (croissance
    logistique, Von Bertalanffy, ...) sans imposer de structure de données
    commune aux résultats, qui diffèrent fortement d'un modèle à l'autre
    (série temporelle 0D pour Gompertz, champ 2D pour Fisher-KPP).
    """

    #: Nom court identifiant le modèle, redéfini par chaque sous-classe.
    name: ClassVar[str] = "base"

    @abstractmethod
    def simulate(self) -> Any:
        """Exécute la simulation avec les paramètres stockés à la construction."""
        raise NotImplementedError
