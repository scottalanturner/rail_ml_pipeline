class PredictionPipeline:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
    
    def predict(self, input_data):
        """Generate predictions using the full pipeline"""
        features = self.processor.transform(input_data)
        predictions = []
        
        for model in self.model.models:
            prob = model.predict(features)
            predictions.append(np.clip(prob, 0, 1))
            
        return self.format_predictions(predictions)
    
    def format_predictions(self, predictions):
        """Format predictions into 10-minute intervals"""
        intervals = [f"{i*10}-{(i+1)*10}" for i in range(6)]
        return dict(zip(intervals, predictions)) 