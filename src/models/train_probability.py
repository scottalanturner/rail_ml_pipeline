import xgboost as xgb
import numpy as np

class TrainProbabilityModel:
    def __init__(self):
        self.models = []
        self.params = {
            'objective': 'reg:logistic',  # For probability output
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1
        }
    
    def train(self, X, y):
        """Train 6 separate models for each interval"""
        for interval in range(6):
            model = xgb.XGBRegressor(**self.params)
            model.fit(X, y[f'interval_{interval}'])
            self.models.append(model)
    
    def predict(self, X):
        """Generate predictions for all intervals"""
        predictions = []
        for model in self.models:
            prob = model.predict(X)
            predictions.append(np.clip(prob, 0, 1))
        return predictions 