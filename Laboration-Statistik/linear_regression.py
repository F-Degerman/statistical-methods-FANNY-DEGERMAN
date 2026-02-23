import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self, alpha=0.05): # konfidensintervall = 95% 
        self.alpha = alpha

        # output
        self.b = None           # vektor av skattade koefficienter (inklusive intercept)
        self.R2 = None          # förklaringsgraden 

        # behövs för tester
        self.cov = None         # kovariansmatris för beta
        self.sigma2 = None      # skattning av residualvariansen
        self.n = None           # antal observationer
        self.d = None           # antal prediktorer (exkl. intercept)
        self.df = None          # frihetsgrader för residualer (n - (d+1))

        # behövs för F-test och R²
        self.SSE = None         # residualsumma - "fel" som modellen inte förklarar (sum of squared errors) 
        self.Syy = None         # total variation i y (total sum of squares)   
        self.SSR = None         # Förklarad variation av modellen (sum of squares regression)
        

    def fit(self, X, y):
        # Regressionen implementeras med numpy-matriser, därför konverteras X och y till numpy-array. 
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.n, p = X.shape     
        self.d = p - 1                      
        self.df = self.n - p               

        XtX_inv = np.linalg.pinv(X.T @ X)    # pseudo-invers av X'X för att hantera singularitet (multikollinearitet) 
        self.b = XtX_inv @ (X.T @ y)

        y_hat = X @ self.b                  # predikterade värden
        r = y - y_hat                       # residualer

        self.SSE = float(np.sum(r**2))      
        y_mean = float(np.mean(y))          
        self.Syy = float(np.sum((y - y_mean)**2))
        self.SSR = self.Syy - self.SSE

        self.sigma2 = self.SSE / self.df if self.df > 0 else np.nan  
        self.cov = XtX_inv * self.sigma2 if np.isfinite(self.sigma2) else np.full_like(XtX_inv, np.nan) 

        self.R2 = self.SSR / self.Syy if self.Syy > 0 else np.nan 
        return self
    
    # standardavvikelse av residualerna (modellens "fel")   
    def std(self):
        return float(np.sqrt(self.sigma2))

    # F-test för att utvärdera modellens signifikans
    def f_test(self):
        # H0: alla betas (utom intercept) = 0
        if self.d <= 0 or not np.isfinite(self.sigma2) or self.sigma2 == 0:
            return np.nan, np.nan
        F = (self.SSR / self.d) / self.sigma2
        p = stats.f.sf(F, self.d, self.df)
        return float(F), float(p)

    # t-test för varje koefficient (inklusive intercept)
    def t_tests(self):
        se = np.sqrt(np.diag(self.cov))
        t = self.b / se
        p = 2 * stats.t.sf(np.abs(t), df=self.df)
        return {"se": se, "t": t, "p": p}

    # Konfidensintervall för varje koefficient (inklusive intercept)
    def confidence_intervals(self):
        se = np.sqrt(np.diag(self.cov))
        tcrit = stats.t.isf(self.alpha / 2, df=self.df)
        low = self.b - tcrit * se
        high = self.b + tcrit * se
        return low, high
    
        # Prediktion av nya värden
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.b

    # variansen av residualerna (modellens "fel")
    def variance(self):
        return float(self.sigma2)

    # Modellens prediktionsfel (MSE och RMSE)   
    def mse(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        e = y - self.predict(X)
        return float(np.mean(e**2))
    def rmse(self, X, y):
        return float(np.sqrt(self.mse(X, y)))

    # Datakontroll med Pearson-korrelation mellan prediktorerna (för att upptäcka multikollinearitet)
    def pearson_X(self, X, include_intercept=False):
        X = np.asarray(X, dtype=float)
        if not include_intercept:
            X = X[:, 1:]
        return np.corrcoef(X, rowvar=False)
