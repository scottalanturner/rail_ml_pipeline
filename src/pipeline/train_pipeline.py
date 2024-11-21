import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException

def train_pipeline():
    try:
        # Initialize components
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()
        model_trainer = ModelTrainer()

        # Ingest data
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Transform data
        train_features, test_features, _ = data_transformation.initiate_data_transformation(
            train_path,
            test_path
        )
        
        # Train the model
        metrics = model_trainer.initiate_model_training(
            train_features, 
            test_features
        )
        
        return metrics

    except Exception as e:
        raise CustomException(e, sys) from e
