from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

def get_simple_prediction():
    """Simple prediction logic:
    - If a train passed in the last hour: probability is near 0
    - If no train in over an hour: probability is high
    """
    try:
        # Read the train data
        df = pd.read_csv('artifacts/train.csv')
        df['time_start'] = pd.to_datetime(df['time_start'])
        
        # Get the most recent train
        most_recent = df.iloc[-1]
        current_time = datetime.now()
        time_since_last = current_time - most_recent['time_start']
        
        # Simple prediction logic
        if time_since_last < timedelta(hours=1):
            probabilities = [0.1] * 6  # Low probability for all intervals
        else:
            probabilities = [0.8] * 6  # High probability for all intervals
            
        # Format for 10-minute intervals
        intervals = [f"{i*10}-{(i+1)*10}" for i in range(6)]
        predictions = dict(zip(intervals, probabilities))
        
        return {
            'predictions': predictions,
            'last_train': {
                'time': most_recent['time_start'].strftime('%H:%M'),
                'time_ago': f"{int(time_since_last.total_seconds() / 60)} minutes ago"
            }
        }
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get prediction
        pred_pipeline = PredictPipeline()
        df = pd.DataFrame({
            'time_start': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        result = pred_pipeline.predict(df)
        print(result)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000) 