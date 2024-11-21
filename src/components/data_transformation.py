import sys
import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation settings and file paths"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """Handles all data transformation operations including feature engineering and preprocessing"""
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.scaler = StandardScaler()

    def get_data_transformer_object(self):
        '''
        This function creates and returns the data transformation object
        '''
        try:
            return {
                'scaler': self.scaler
            }
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def compute_time_since_last_train(self, df):
        """Calculate minutes since the last train crossing"""
        try:
            # Sort by timestamp to ensure correct ordering
            df = df.sort_values('timestamp')
            
            # Calculate time difference in minutes
            time_diff = df['timestamp'].diff()
            minutes_since_last = time_diff.dt.total_seconds() / 60
            
            return minutes_since_last.fillna(0)  # Fill first value with 0
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def compute_rolling_train_count(self, df, window):
        """Calculate number of trains in the specified window"""
        try:
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Create a Series of 1s (representing each train)
            train_series = pd.Series(1, index=df['timestamp'])
            
            # Calculate rolling count for each timestamp
            rolling_count = train_series.rolling(
                window=window,
                min_periods=1
            ).sum()
            
            return rolling_count.values
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def compute_historical_average(self, df, group_by):
        """Calculate average number of trains in the specified group"""
        try:
            if group_by == 'hour':
                # Extract hour from timestamp
                df = df.copy()
                df['hour'] = df['timestamp'].dt.hour
                avg_trains = df.groupby('hour').size().reindex(range(24), fill_value=0)
                # Map back to original dataframe
                return df['hour'].map(avg_trains)
                
            elif group_by == 'dayofweek':
                # Extract day of week from timestamp
                df = df.copy()
                df['dayofweek'] = df['timestamp'].dt.dayofweek
                avg_trains = df.groupby('dayofweek').size().reindex(range(7), fill_value=0)
                # Map back to original dataframe
                return df['dayofweek'].map(avg_trains)
            else:
                raise ValueError(f"Unsupported grouping: {group_by}")
                
        except Exception as e:
            raise CustomException(e, sys) from e

    def create_features(self, df):
        """Create features for survival analysis"""
        try:
            logging.info("Started feature creation")
            
            features = pd.DataFrame()
            
            # Time-based features
            features['hour_of_day'] = df['timestamp'].dt.hour
            features['is_daytime'] = (df['timestamp'].dt.hour >= 6) & \
                                   (df['timestamp'].dt.hour <= 18)
            
            # Calculate time until next train (survival time)
            features['time_until_next'] = df['timestamp'].shift(-1) - df['timestamp']
            features['time_until_next'] = features['time_until_next'].dt.total_seconds() / 60
            
            # Add small positive value (0.1 minutes) to zero durations
            features['time_until_next'] = features['time_until_next'].clip(lower=0.1)
            
            # Event indicator (1 if we observe the next train, 0 if censored)
            features['event'] = 1
            features.loc[features.index[-1], 'event'] = 0  # Last observation is censored
            
            # For the last row, set time_until_next to a censored value
            max_time = features['time_until_next'].max()
            features.loc[features.index[-1], 'time_until_next'] = max_time
            
            # Time since last train
            features['minutes_since_last'] = self.compute_time_since_last_train(df)
            features['minutes_since_last'] = features['minutes_since_last'].clip(lower=0.1)  # Ensure positive
            
            # Rolling window features
            features['trains_last_4_hours'] = self.compute_rolling_train_count(df, '4h')
            features['trains_last_1_hour'] = self.compute_rolling_train_count(df, '1h')
            features['trains_last_2_hours'] = self.compute_rolling_train_count(df, '2h')
            
            # Cyclical time encoding
            features['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
            
            # Additional time-based features
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
            features['month'] = df['timestamp'].dt.month
            features['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
            
            # Historical patterns
            features['avg_trains_this_hour'] = self.compute_historical_average(df, 'hour')
            features['avg_trains_this_dow'] = self.compute_historical_average(df, 'dayofweek')
            
            # Scale numerical features
            numerical_features = [
                'hour_of_day', 'minutes_since_last', 'trains_last_4_hours',
                'trains_last_1_hour', 'trains_last_2_hours', 'minutes_since_midnight',
                'avg_trains_this_hour', 'avg_trains_this_dow'
            ]
            features[numerical_features] = self.scaler.fit_transform(features[numerical_features])
            
            # Verify no NaN values exist
            if features.isnull().any().any():
                null_columns = features.columns[features.isnull().any()].tolist()
                raise ValueError(f"NaN values found in columns: {null_columns}")
            
            logging.info("Completed feature creation")
            return features
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def create_time_varying_features(self, df):
        """Create features that change with time"""
        features = []
        
        for row in df.itertuples():
            base_time = row.timestamp
            future_times = [base_time + pd.Timedelta(minutes=i*10) for i in range(7)]
            
            for future_time in future_times:
                row_features = {
                    'start_hour': base_time.hour,
                    'future_hour': future_time.hour,
                    'minutes_elapsed': (future_time - base_time).total_seconds() / 60,
                    'is_daytime': (future_time.hour >= 6) and (future_time.hour <= 18)
                }
                features.append(row_features)
        
        return pd.DataFrame(features)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Main method to initiate the transformation process
        """
        try:
            logging.info("Started data transformation")
            
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Convert timestamp columns
            train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            
            # Create features
            train_features = self.create_features(train_df)
            test_features = self.create_features(test_df)
            
            # Save preprocessor object
            preprocessor = self.get_data_transformer_object()
            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            logging.info("Completed data transformation")
            
            return (
                train_features,
                test_features,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys) from e
        
if __name__ == "__main__":
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Initialize transformation object
        obj = DataTransformation()
        
        # Execute transformation pipeline with absolute paths
        train_data, test_data, preprocessor_path = obj.initiate_data_transformation(
            os.path.join(project_root, 'artifacts', 'train.csv'),
            os.path.join(project_root, 'artifacts', 'test.csv')
        )
        
        # Print basic validation info
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Preprocessor saved at: {preprocessor_path}")
        
        # Verify features were created correctly
        expected_columns = [
            'hour_of_day', 'is_daytime', 'hour_sin', 'hour_cos',
            'minutes_since_last', 'trains_last_4_hours'
        ]
        assert all(col in train_data.columns for col in expected_columns), "Missing expected columns"
        
        print("Data transformation completed successfully!")
        
    except Exception as e:
        print(f"Error during transformation: {e}")