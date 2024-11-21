import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index
from src.logger import logging

class SurvivalProbabilityModel:
    def __init__(self, model_type='cox'):
        """Initialize the survival model
        
        Args:
            model_type (str): Type of survival model to use
                            Options: 'cox', 'weibull', 'lognormal', 'loglogistic'
        """
        self.model_type = model_type
        if model_type == 'cox':
            self.model = CoxPHFitter()
        elif model_type == 'weibull':
            self.model = WeibullAFTFitter()
        elif model_type == 'lognormal':
            self.model = LogNormalAFTFitter()
        elif model_type == 'loglogistic':
            self.model = LogLogisticAFTFitter()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def train(self, features):
        """Train the survival model"""
        try:
            # Reduced feature set to avoid collinearity
            train_cols = [
                'hour_of_day',          # Keep basic time feature
                'is_daytime',           # Important binary feature
                'minutes_since_last',    # Key timing feature
                'trains_last_4_hours',   # Keep longest window
                'hour_sin',             # Keep cyclical encoding
                'hour_cos',
                'is_weekend',           # Important binary feature
                'avg_trains_this_hour', # Keep most relevant average
                'time_until_next',      # Required for survival analysis
                'event'                 # Required for survival analysis
            ]
            
            # Remove features with very low variance
            feature_df = features[train_cols].copy()
            variances = feature_df.var()
            low_var_features = variances[variances < 0.01].index.tolist()
            
            # Remove low variance features except time_until_next and event
            for feature in low_var_features:
                if feature not in ['time_until_next', 'event']:
                    train_cols.remove(feature)
            
            logging.info(f"Training with features: {train_cols}")
            
            self.model.fit(
                features[train_cols],
                duration_col='time_until_next',
                event_col='event'
            )
            
            return self.model
            
        except Exception as e:
            raise e