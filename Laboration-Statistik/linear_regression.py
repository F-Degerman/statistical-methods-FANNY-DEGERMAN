import numpy as np
from scipy import stats

class LinearRegression:
    """
    Multipel linjär regression
    """
    def __init__(self, alpha=0.05):     # 0.5 betyder 95% konfidensintervall 

        self.alpha = alpha              #för att tolka resultaten statistiskt


    def fit(self, X, y):
        """
        Skattar beta med OLS:
        b = (X^T X)^(-1) X^T y
        """
        # Lägg till en kolumn av 1:or för intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Beräkna beta med OLS formel
        self.b_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        # Spara antal observationer och prediktorer
        self.n_sample_size = X.shape[0]
        self.n_predictors = X.shape[1]

    def predict(self, X):
        """
        Returnerar predikterade värden för X
        """
        # Lägg till en kolumn av 1:or för intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Returnera predikterade värden
        return X_with_intercept @ self.b_hat

    def rmse(self, X, y):   
        """
        Beräknar Root Mean Squared Error (RMSE) för modellen
        """
        y_pred = self.predict(X)
        residuals = y - y_pred
        rmse_value = np.sqrt(np.mean(residuals**2))
        return rmse_value   


    def f_test(self):   
        # F-test för att testa den övergripande signifikansen av modellen
            self.f_statistic = (self.ssr / self.n_predictors) / (self.sse / (self.n_sample_size - self.n_predictors - 1))
            self.f_p_value = 1 - stats.f.cdf(self.f_statistic, self.n_predictors, self.n_sample_size - self.n_predictors - 1)   


    def t_tests(self):
        # t-test för att testa signifikansen av varje enskild prediktor
        self.t_statistics = self.b_hat / np.sqrt(self.sse / (self.n_sample_size - self.n_predictors - 1) * np.diag(np.linalg.inv(X.T @ X)))
        self.t_p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_statistics), df=self.n_sample_size - self.n_predictors - 1))   

    
    def confidence_intervals(self):
        # Konfidensintervall för varje koefficient
        critical_value = stats.t.ppf(1 - self.alpha / 2, df=self.n_sample_size - self.n_predictors - 1)
        standard_errors = np.sqrt(self.sse / (self.n_sample_size - self.n_predictors - 1) * np.diag(np.linalg.inv(X.T @ X)))
        self.confidence_intervals = np.column_stack((self.b_hat - critical_value * standard_errors, self.b_hat + critical_value * standard_errors)) 

    
    

