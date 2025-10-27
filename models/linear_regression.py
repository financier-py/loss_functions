import numpy as np
from loss_functions.base import BaseLoss
from optimizers.base import BaseOptimizer


class LinearRegression:
    """Линейная регрессия с различными функциями потерь и методами оптимизации."""

    def __init__(self, loss_function: BaseLoss, optimizer: BaseOptimizer, max_iter: int = 1000, tol: float = 1e-6):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol = tol

        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Обучение модели

        Args:
            X (np.ndarray): shape (n_samples, n_features) - матрица признаков.
            y (np.ndarray): shape (n_samples,) - целевая переменная.
        Returns:
            self: 'LinearRegression'
        """

        n_samples, n_features = X.shape

        if self.weights is None:
            self.weights = np.zeros(n_features)

        self._fit_gradient_descent(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' before 'predict'.")
        return X @ self.weights

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Обучение через градиентный спуск."""

        for iteration in range(self.max_iter):
            predictions = self.predict(X)
            gradient = self.loss_function.gradient(y, predictions, X)

            new_weights = self.optimizer.step(gradient, self.weights)

            if np.linalg.norm(new_weights - self.weights) < self.tol:
                break

            self.weights = new_weights