import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.models.train_probability import TrainProbabilityModel
from src.data.processor import TrainDataProcessor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model = TrainProbabilityModel()
        self.processor = TrainDataProcessor()

    def initiate_model_training(self, train_data, test_data):
        try:
            logging.info("Splitting training and testing input data")
            
            # Process features
            X_train = self.processor.transform(train_data)
            X_test = self.processor.transform(test_data)
            
            # Create target variables for each 10-minute interval
            y_train = self.processor.create_targets(train_data)
            y_test = self.processor.create_targets(test_data)

            # Train the model (this trains 6 separate XGBoost models)
            self.model.train(X_train, y_train)

            # Evaluate on test data
            test_predictions = self.model.predict(X_test)
            
            # Calculate metrics for each interval
            metrics = {}
            for interval in range(6):
                interval_name = f"{interval*10}-{(interval+1)*10}_minutes"
                mse = mean_squared_error(y_test[f'interval_{interval}'], 
                                       test_predictions[interval])
                r2 = r2_score(y_test[f'interval_{interval}'], 
                             test_predictions[interval])
                
                metrics[interval_name] = {
                    'mse': mse,
                    'r2': r2
                }
                
                logging.info(f"Metrics for {interval_name}:")
                logging.info(f"MSE: {mse}")
                logging.info(f"R2 Score: {r2}")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)
