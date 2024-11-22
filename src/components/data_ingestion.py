import os
import sys
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Started data ingestion")
            
            # Read the dataset
            df = pd.read_csv('notebook/data/train_positions.csv')
            df['timestamp'] = pd.to_datetime(df['time_start'])
            
            # Sort by timestamp to ensure temporal order
            df = df.sort_values('timestamp')
            
            logging.info("Read dataset as dataframe")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), 
                       exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Time-based split (last 20% for testing)
            split_idx = int(len(df) * 0.8)
            train_set = df[:split_idx]
            test_set = df[split_idx:]

            logging.info("Train-test split completed")
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, 
                           index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, 
                           index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e
             
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(train_data, test_data)
