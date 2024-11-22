import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def transform_data(df):
        """Static method for data transformation"""
        try:
            # Convert timestamps
            df['time_start'] = pd.to_datetime(df['time_start'])
            
            # Extract time features
            df['hour'] = df['time_start'].dt.hour
            df['minute'] = df['time_start'].dt.minute
            df['day_of_week'] = df['time_start'].dt.dayofweek
            df['is_daytime'] = (df['hour'] >= 6) & (df['hour'] <= 18)
            
            # Calculate time since last train
            df['minutes_since_last'] = df['time_start'].diff().dt.total_seconds() / 60
            df['minutes_since_last'] = df['minutes_since_last'].fillna(60)  # First entry
            
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """Return the static transform method"""
        return self.transform_data

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            # Get data transformer
            transformer = self.get_data_transformer_object()
            
            # Transform data
            train_df = transformer(train_df)
            test_df = transformer(test_df)
            
            # Select features for training
            feature_cols = ['hour', 'day_of_week', 'is_daytime', 'minutes_since_last']
            
            train_arr = train_df[feature_cols].values
            test_arr = test_df[feature_cols].values
            
            logging.info("Saving preprocessor")
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=transformer
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path
            )
            
        except Exception as e:
            raise CustomException(e, sys) from e

def main():
    try:
        # Initialize data transformation
        data_transform = DataTransformation()
        
        # Test paths
        train_path = os.path.join('artifacts', 'train.csv')
        test_path = os.path.join('artifacts', 'test.csv')
        
        # Verify files exist
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("Train or test data not found. Run data ingestion first.")
            return
            
        # Transform data
        train_arr, test_arr, preprocessor_path = data_transform.initiate_data_transformation(
            train_path=train_path,
            test_path=test_path
        )
        
        # Print validation info
        print("Training data shape:", train_arr.shape)
        print("Test data shape:", test_arr.shape)
        print("Preprocessor saved at:", preprocessor_path)
        
        # Test loading and using the transformer

        transformer = load_object(preprocessor_path)
            
        # Test transformer on sample data
        sample_df = pd.DataFrame({
            'time_start': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        transformed = transformer(sample_df)
        print("\nSample transformation:")
        print(transformed.head())
        
    except Exception as e:
        print(f"Error during transformation: {e}")

if __name__ == "__main__":
    main()