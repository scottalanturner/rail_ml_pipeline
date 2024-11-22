"""Pipeline for generating predictions using trained models."""
import sys
import os
import logging
import pandas as pd
import pickle
from datetime import datetime
import numpy as np
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load preprocessor and model using utility function
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Apply preprocessor function
            print("Transforming features...")
            transformed_df = preprocessor(features)
            
            # Select the same features used in training
            feature_cols = ['hour', 'day_of_week', 'is_daytime', 'minutes_since_last']
            data_scaled = transformed_df[feature_cols].values
            
            # Get prediction probabilities
            print("Making prediction...")
            predictions = model.predict_proba(data_scaled)
            
            # Format predictions for 10-minute intervals
            intervals = [f"{i*10}-{(i+1)*10}" for i in range(6)]
            # Use the probability of class 1 (train within 60 minutes)
            probs = [round(float(predictions[0][1]) * 100, 1)] * 6  # Convert to percentage and round
            print('***********')
            print(f'minutes since last: {transformed_df["minutes_since_last"].iloc[0]}')
            print('***********')
            # Return formatted result
            result = {
                'predictions': dict(zip(intervals, probs)),
                'last_train': {
                    'time': transformed_df['time_start'].iloc[0].strftime('%H:%M'),
                    'minutes': str(transformed_df['minutes_since_last'].iloc[0])
                }
            }
            print("Prediction result:", result)
            return result
            
        except Exception as e:
            print(f"Exception in prediction pipeline: {e}")
            raise e

class CustomData:
    def __init__(self, timestamp: str):
        self.timestamp = timestamp

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "timestamp": [self.timestamp]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
        
            return df
        
        except Exception as e:
            raise CustomException(e, sys) from e

def main():
    try:
        # Create sample input
        current_time = datetime.now()
        df = pd.DataFrame({
            'time_start': [current_time.strftime('%Y-%m-%d %H:%M:%S')]
        })
        print(df)
        
        # Initialize pipeline and predict
        pred_pipeline = PredictPipeline()
        preds = pred_pipeline.predict(df)
        
        print("\nPredictions:")
        for interval, prob in preds['predictions'].items():
            print(f"{interval} minutes: {prob:.1%} chance of train")
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()