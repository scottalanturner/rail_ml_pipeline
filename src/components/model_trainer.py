import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import timedelta

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def create_targets(self, df):
        """Create target values based on time between trains"""
        df['time_start'] = pd.to_datetime(df['time_start'])
        df['next_train'] = df['time_start'].shift(-1)
        df['minutes_to_next'] = (df['next_train'] - df['time_start']).dt.total_seconds() / 60
        
        # Target is 1 if next train is within 60 minutes
        df['target'] = (df['minutes_to_next'] <= 60).astype(int)
        
        return df['target'].values
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training")
            
            # Read original data to get timestamps
            train_data = pd.read_csv(os.path.join('artifacts', 'train.csv'))
            
            # Create meaningful targets
            y = self.create_targets(train_data)
            X = train_array[:len(y)]  # Match features to targets
            
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Positive samples (train within 60 min): {sum(y)}")
            print(f"Negative samples (no train within 60 min): {len(y) - sum(y)}")
            
            # Split training data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train random forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Print model performance
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"\nTrain accuracy: {train_score:.3f}")
            print(f"Test accuracy: {test_score:.3f}")
            
            # Save the model
            with open(self.model_trainer_config.trained_model_path, 'wb') as f:
                pickle.dump(model, f)
                
            logging.info("Model training completed")
            return self.model_trainer_config.trained_model_path
            
        except Exception as e:
            raise CustomException(e, sys)

def main():
    try:
        # Get transformed data
        train_path = os.path.join('artifacts', 'train.csv')
        test_path = os.path.join('artifacts', 'test.csv')
        
        from src.components.data_transformation import DataTransformation
        data_transformation = DataTransformation()
        
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        
        # Train model
        modeltrainer = ModelTrainer()
        model_path = modeltrainer.initiate_model_trainer(train_arr, test_arr)
        print(f"\nModel saved to: {model_path}")
        
        # Load and test the model
        print("\nTesting model loading...")
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            
        print("\nModel details:")
        print(f"Type: {type(loaded_model)}")
        print(f"Number of trees: {loaded_model.n_estimators}")
        
        # Feature importance
        feature_names = ['hour', 'day_of_week', 'is_daytime', 'minutes_since_last']
        importances = zip(feature_names, loaded_model.feature_importances_)
        print("\nFeature importances:")
        for name, importance in sorted(importances, key=lambda x: x[1], reverse=True):
            print(f"{name}: {importance:.3f}")
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()