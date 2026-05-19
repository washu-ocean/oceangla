from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import numpy as np

from .cluster import ClusterCorrectionMixin


class GroupLevelModelResults(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass


class GroupLevelModel(ABC):
    @abstractmethod
    def __init__(self,
                 design_matrix: pd.DataFrame,
                 activation: dict,
                 model_desc: str,
                 perms: int = 0,
                 alpha: float = 0.05,
                 l_surf_path: Path = None,
                 r_surf_path: Path = None,
                 l_area_path: Path = None,
                 r_area_path: Path = None,
                 **kwargs):
        self.design_matrix = design_matrix
        self.activation = activation
        self.model_desc = model_desc
        self.perms = perms
        self.alpha = alpha
        self.l_surf_path = l_surf_path
        self.r_surf_path = r_surf_path
        self.l_area_path = l_area_path
        self.r_area_path = r_area_path

    @classmethod
    def cluster_correct(cls, pvalues):
        pass

    @abstractmethod
    def get_activation(self):
        pass

    @abstractmethod
    def permute_design_matrix(self):
        pass

    @abstractmethod
    def fit(self) -> GroupLevelModelResults:
        pass

    @abstractmethod
    def _fit_permutation(self) -> GroupLevelModelResults:
        pass
    
    @abstractmethod
    def _get_clusters_from_pvalue_map(self):
        pass
