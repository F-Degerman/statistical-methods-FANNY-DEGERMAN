import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

        # Fit-resultat (behövs av din notebook)
        self.b = None
        self.cov = None
        self.sigma2 = None
        self.R2 = None

        # Frihetsgrader (behövs för tester)
        self.n = None
        self.d = None
        self.df = None

        # Summor (behövs för F-test och R2)
        self.SSE = None
        self.SSR = None
        self.Syy = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.n, p = X.shape          # p inkluderar intercept-kolumnen
        self.d = p - 1               # antal prediktorer exkl intercept
        self.df = self.n - p         # = n - (d+1)

        # OLS: b = (X^T X)^(-1) X^T y  (pseudoinvers för robusthet)
        XtX_inv = np.linalg.pinv(X.T @ X)
        self.b = XtX_inv @ (X.T @ y)

        y_hat = X @ self.b
        residuals = y - y_hat

        self.SSE = float(np.sum(residuals**2))
        y_mean = float(np.mean(y))
        self.Syy = float(np.sum((y - y_mean) ** 2))
        self.SSR = self.Syy - self.SSE

        self.sigma2 = self.SSE / self.df if self.df > 0 else np.nan
        self.cov = XtX_inv * self.sigma2 if np.isfinite(self.sigma2) else np.full_like(XtX_inv, np.nan)

        self.R2 = self.SSR / self.Syy if self.Syy > 0 else np.nan
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.b

    def mse(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        e = y - self.predict(X)
        return float(np.mean(e**2))

    def rmse(self, X, y):
        return float(np.sqrt(self.mse(X, y)))

    def std(self):
        return float(np.sqrt(self.sigma2))

    def regression_significance(self):
        # H0: alla beta (utom intercept) = 0
        if self.d is None or self.d <= 0 or not np.isfinite(self.sigma2) or self.sigma2 == 0:
            return np.nan, np.nan

        F_stat = (self.SSR / self.d) / self.sigma2
        p_val = stats.f.sf(F_stat, self.d, self.df)
        return float(F_stat), float(p_val)

    def parameter_tests(self):
        se = np.sqrt(np.diag(self.cov))
        t_stat = self.b / se

        # tvåsidigt p-värde
        p_vals = 2 * stats.t.sf(np.abs(t_stat), df=self.df)
        return {"se": se, "t": t_stat, "p": p_vals}

    def confidence_intervals(self):
        tests = self.parameter_tests()
        se = tests["se"]
        t_crit = stats.t.isf(self.alpha / 2, df=self.df)

        low = self.b - t_crit * se
        high = self.b + t_crit * se
        return low, high

    def pearson_matrix(self, X, include_intercept=False):
        X = np.asarray(X, dtype=float)
        if not include_intercept:
            X = X[:, 1:]
        return np.corrcoef(X, rowvar=False)
