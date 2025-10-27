from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
    """Абстрактный базовый класс для оптимизаторов."""

    def __init__(self, learning_rate: float = 0.01):
        """ learning_rate (float): Скорость обучения."""
        self.learning_rate = learning_rate
        self.iteration = 0

    def step(self, gradient: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Выполняет один шаг оптимизации

        Args:
            gradient (np.ndarray): градиент функции потерь.
            weights (np.ndarray): текущие веса.
        Returns:
            np.ndarray: обновленные веса.
        """
        pass

    def reset(self):
        """Сброс состояния оптимизатора."""
        self.iteration = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Возвращает имя оптимизатора."""
        pass
