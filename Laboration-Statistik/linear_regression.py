import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

        # output
        self.b = None
        self.R2 = None

        # behövs för tester
        self.cov = None
        self.sigma2 = None
        self.n = None
        self.d = None
        self.df = None

        # behövs för F-test och R²
        self.SSE = None
        self.SSR = None
        self.Syy = None

    def fit(self, X, y):
        # Regressionen implementeras med numpy-matriser, därför konverteras X och y till numpy-array. 
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.n, p = X.shape
        self.d = p - 1
        self.df = self.n - p  # n - (d+1)

        XtX_inv = np.linalg.pinv(X.T @ X)   # robustare än inv()
        self.b = XtX_inv @ (X.T @ y)

        y_hat = X @ self.b
        r = y - y_hat

        self.SSE = float(np.sum(r**2))
        y_mean = float(np.mean(y))
        self.Syy = float(np.sum((y - y_mean)**2))
        self.SSR = self.Syy - self.SSE

        self.sigma2 = self.SSE / self.df if self.df > 0 else np.nan
        self.cov = XtX_inv * self.sigma2 if np.isfinite(self.sigma2) else np.full_like(XtX_inv, np.nan)

        self.R2 = self.SSR / self.Syy if self.Syy > 0 else np.nan
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.b

    def variance(self):
        return float(self.sigma2)

    def std(self):
        return float(np.sqrt(self.sigma2))

    def f_test(self):
        # H0: alla betas (utom intercept) = 0
        if self.d <= 0 or not np.isfinite(self.sigma2) or self.sigma2 == 0:
            return np.nan, np.nan
        F = (self.SSR / self.d) / self.sigma2
        p = stats.f.sf(F, self.d, self.df)
        return float(F), float(p)

    def t_tests(self):
        se = np.sqrt(np.diag(self.cov))
        t = self.b / se
        p = 2 * stats.t.sf(np.abs(t), df=self.df)
        return {"se": se, "t": t, "p": p}

    def confidence_intervals(self):
        se = np.sqrt(np.diag(self.cov))
        tcrit = stats.t.isf(self.alpha / 2, df=self.df)
        low = self.b - tcrit * se
        high = self.b + tcrit * se
        return low, high

    def mse(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        e = y - self.predict(X)
        return float(np.mean(e**2))

    def rmse(self, X, y):
        return float(np.sqrt(self.mse(X, y)))

    def pearson_X(self, X, include_intercept=False):
        X = np.asarray(X, dtype=float)
        if not include_intercept:
            X = X[:, 1:]
        return np.corrcoef(X, rowvar=False)
