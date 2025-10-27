from abc import ABC, abstractmethod
import numpy as np

class BaseLoss(ABC):
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисляет значение функции потерь."""
        pass

    @abstractmethod
    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Вычисляет градиент функции потерь."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Возвращает имя функции потерь."""
        pass