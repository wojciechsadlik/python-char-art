from abc import ABC, abstractmethod
import numpy as np


class TileConverter(ABC):
    @abstractmethod
    def tile2symbol(self, tile: np.ndarray) -> str:
        pass
