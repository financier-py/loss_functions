from abc import ABC, abstractmethod
import numpy as np

class BaseLoss(ABC):
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисляет значение функции потерь."""
        pass

    @abstractmethod
    def compute_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Вычисляет градиент функции потерь."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Возвращает имя функции потерь."""
        pass

class MSELoss(BaseLoss):
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def compute_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        errors = y_pred - y_true
        gradient = (2 / n_samples) * (X.T @ errors)
        return gradient

    @property
    def name(self) -> str:
        return "MSE"

class MAELoss(BaseLoss):
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def compute_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        errors = y_pred - y_true
        gradient = (1 / n_samples) * (X.T @ np.sign(errors))
        return gradient

    @property
    def name(self) -> str:
        return "MAE"