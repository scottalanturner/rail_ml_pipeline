from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer

def train_pipeline():
    try:
        # Initialize components
        data_ingestion = DataIngestion()
        model_trainer = ModelTrainer()

        # Ingest and split data
        train_path, test_path = data_ingestion.initiate_data_ingestion(
            "path/to/your/data.csv"
        )
        
        # Read the split data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Train the model
        metrics = model_trainer.initiate_model_training(train_data, test_data)
        
        return metrics

    except Exception as e:
        raise CustomException(e, sys)
