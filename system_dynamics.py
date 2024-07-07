import numpy as np


class VanDerPol:
    def __init__(self, delta_t=0.01, mu=2, alpha=5, beta=0.8) -> None:
        self.delta_t = delta_t
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def f(self, x):
        return np.array(
            [
                2 * x[1],
                self.mu * (1 - self.alpha * x[0] * x[0]) * x[1] - self.beta * x[0],
            ]
        )

    def f_u(self, x, u):
        return self.f(x) + np.array([0, u])

    def k1(self, x, u):
        return self.f_u(x, u)

    def k2(self, x, u):
        return self.f_u(x + self.k1(x, u) * self.delta_t / 2, u)

    def k3(self, x, u):
        return self.f_u(x + self.k2(x, u) * self.delta_t / 2, u)

    def k4(self, x, u):
        return self.f_u(x + self.k1(x, u) * self.delta_t, u)

    def f_ud(self, x, u):
        return x + (self.delta_t / 6) * (
            self.k1(x, u) + 2 * self.k2(x, u) + 2 * self.k3(x, u) + self.k4(x, u)
        )


class Duffing:
    def __init__(self, delta_t=0.01, alpha=1, beta=-1, delta=0.5) -> None:
        self.delta_t = delta_t
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def f(self, x):
        return np.array(
            [
                2 * x[1],
                -self.delta * 2 * x[1]
                - 2 * x[0] * (self.beta + self.alpha * (2 * x[0]) ** 2),
            ]
        )

    def f_u(self, x, u):
        return self.f(x) + np.array([0, u])

    def k1(self, x, u):
        return self.f_u(x, u)

    def k2(self, x, u):
        return self.f_u(x + self.k1(x, u) * self.delta_t / 2, u)

    def k3(self, x, u):
        return self.f_u(x + self.k2(x, u) * self.delta_t / 2, u)

    def k4(self, x, u):
        return self.f_u(x + self.k1(x, u) * self.delta_t, u)

    def f_ud(self, x, u):
        return x + (self.delta_t / 6) * (
            self.k1(x, u) + 2 * self.k2(x, u) + 2 * self.k3(x, u) + self.k4(x, u)
        )
