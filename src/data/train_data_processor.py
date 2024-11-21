import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

class TrainDataProcessor:
    """Processes train data to create features and target variables for model training."""
    def __init__(self):
        self.feature_columns = [
            'hour_sin', 'hour_cos', 'is_daytime',
            'minutes_since_last_train',
            'trains_last_4_hours', 'trains_last_12_hours'
        ]
        
    def transform(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw input data into model-ready features
        
        Args:
            input_data (pd.DataFrame): Raw input data with timestamp column
            
        Returns:
            pd.DataFrame: Transformed features
        """
        try:
            logging.info("Starting data transformation")
            
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            if 'timestamp' not in input_data.columns:
                raise ValueError("Input DataFrame must contain 'timestamp' column")
            
            # Ensure timestamp is datetime
            input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
            
            # Return only the required feature columns
            return input_data[self.feature_columns]
            
        except Exception as e:
            logging.error(f"Error in transform method: {str(e)}")
            raise CustomException(e, sys)
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for each 10-minute interval
        
        Args:
            df (pd.DataFrame): Input DataFrame with timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with target variables for each interval
        """
        try:
            logging.info("Creating target variables")
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Initialize target DataFrame
            targets = pd.DataFrame()
            
            # Create 6 intervals (0-10, 10-20, ..., 50-60 minutes)
            for i in range(6):
                interval_start = i * 10
                interval_end = (i + 1) * 10
                
                # Create column name for this interval
                col_name = f'interval_{i}'
                
                # Calculate if a train appears in this interval
                targets[col_name] = self._calculate_interval_probability(
                    df,
                    interval_start,
                    interval_end
                )
            
            logging.info(f"Created {len(targets.columns)} target variables")
            return targets
            
        except Exception as e:
            logging.error(f"Error in create_targets method: {str(e)}")
            raise CustomException(e, sys)
    
    def _calculate_interval_probability(self, df: pd.DataFrame, 
                                     start_minutes: int, 
                                     end_minutes: int) -> pd.Series:
        """
        Calculate the probability of a train appearing in the given interval
        
        Args:
            df (pd.DataFrame): Input DataFrame
            start_minutes (int): Start of interval in minutes
            end_minutes (int): End of interval in minutes
            
        Returns:
            pd.Series: Binary series indicating train presence in interval
        """
        try:
            # Convert minutes to timedelta
            start_td = pd.Timedelta(minutes=start_minutes)
            end_td = pd.Timedelta(minutes=end_minutes)
            
            # For each timestamp, check if a train appears in the interval
            result = pd.Series(index=df.index, dtype=float)
            
            for idx, row in df.iterrows():
                current_time = row['timestamp']
                interval_start = current_time + start_td
                interval_end = current_time + end_td
                
                # Check if any train appears in this interval
                train_in_interval = df[
                    (df['timestamp'] > interval_start) & 
                    (df['timestamp'] <= interval_end)
                ].shape[0] > 0
                
                result[idx] = float(train_in_interval)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in _calculate_interval_probability: {str(e)}")
            raise CustomException(e, sys)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input data meets requirements
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check required columns
            required_columns = ['timestamp'] + self.feature_columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check timestamp column
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                raise ValueError("timestamp column must be datetime type")
            
            # Check for null values
            if df[required_columns].isnull().any().any():
                raise ValueError("Data contains null values")
            
            return True
            
        except Exception as e:
            logging.error(f"Data validation failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test code
    try:
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'hour_sin': np.sin(2 * np.pi * dates.to_series().dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dates.to_series().dt.hour / 24),
            'is_daytime': (dates.to_series().dt.hour >= 6) & (dates.to_series().dt.hour <= 18),
            'minutes_since_last_train': np.random.rand(100) * 60,
            'trains_last_4_hours': np.random.randint(0, 5, 100),
            'trains_last_12_hours': np.random.randint(0, 10, 100)
        })
        
        # Initialize processor
        processor = TrainDataProcessor()
        
        # Test validation
        processor.validate_data(sample_data)
        
        # Test transformation
        features = processor.transform(sample_data)
        
        # Test target creation
        targets = processor.create_targets(sample_data)
        
        print("Features shape:", features.shape)
        print("Targets shape:", targets.shape)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}") 