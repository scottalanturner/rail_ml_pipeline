import os
import sys
import pickle
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.models.survival_probability import SurvivalProbabilityModel
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'survival_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_types = ['cox', 'weibull', 'lognormal', 'loglogistic']
        self.best_model = None
        self.best_model_type = None
        self.best_c_index = 0

    def initiate_model_training(self, train_data, test_data):
        try:
            logging.info("Starting model comparison and training")
            
            model_comparison = {}
            
            # Train and evaluate individual models
            for model_type in self.model_types:
                logging.info(f"\nTraining {model_type.upper()} model...")
                model = SurvivalProbabilityModel(model_type)
                model.train(train_data)
                
                # Calculate metrics for this model
                metrics = self._calculate_metrics(test_data, model)
                model_comparison[model_type] = metrics
                
                # Track best performing model
                if metrics['model_performance']['concordance_index'] > self.best_c_index:
                    self.best_c_index = metrics['model_performance']['concordance_index']
                    self.best_model = model
                    self.best_model_type = model_type
            
            # Save the best model
            with open(self.model_trainer_config.trained_model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logging.info(f"\nBest model ({self.best_model_type}) saved to {self.model_trainer_config.trained_model_path}")
            
            metrics = {
                'model_comparison': model_comparison,
                'best_model_type': self.best_model_type,
                'best_c_index': self.best_c_index
            }
            
            # Format and print readable output
            formatted_output = self._format_metrics(metrics)
            print(formatted_output)
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def _calculate_metrics(self, test_data, model):
        """Calculate survival analysis specific metrics"""
        try:
            metrics = {}
            
            # Concordance index (C-index) - handle different model types
            if isinstance(model.model, CoxPHFitter):
                predictions = -model.model.predict_partial_hazard(test_data)
            else:
                # For parametric models, use predicted expectation
                predictions = model.model.predict_expectation(test_data)
            
            c_index = concordance_index(
                test_data['time_until_next'],
                predictions,
                test_data['event']
            )
            logging.info(f"Concordance Index: {c_index}")
            
            # Get time points for survival function
            times = np.linspace(0, test_data['time_until_next'].max(), 100)
            
            # Get baseline survival function based on model type
            if isinstance(model.model, CoxPHFitter):
                baseline_survival = model.model.baseline_survival_
            else:
                # For parametric models, predict survival function
                baseline_survival = model.model.predict_survival_function(
                    test_data.iloc[0:1],  # Use first row as reference
                    times=times
                )
            
            # Get actual time points from baseline survival
            available_times = baseline_survival.index.values
            
            # Calculate metrics for different time horizons
            for i in range(6):
                interval_name = f"{i*10}-{(i+1)*10}_minutes"
                time_horizon = (i+1)*10
                
                # Find nearest available time point
                nearest_time = available_times[
                    np.abs(available_times - time_horizon).argmin()
                ]
                
                # Get survival probability at nearest time
                if isinstance(model.model, CoxPHFitter):
                    survival_prob = baseline_survival.iloc[
                        baseline_survival.index.get_indexer([nearest_time])[0]
                    ].values[0]
                else:
                    survival_prob = baseline_survival.loc[nearest_time].iloc[0]
                
                metrics[interval_name] = {
                    'baseline_survival_prob': float(survival_prob),
                    'actual_time_point': float(nearest_time)
                }
                
                logging.info(f"Baseline survival probability for {interval_name}: {survival_prob}")
                logging.info(f"Using time point: {nearest_time} minutes")
            
            # Model performance metrics
            metrics['model_performance'] = {
                'concordance_index': c_index,
                'log_likelihood': float(model.model.log_likelihood_)
            }
            
            # Add appropriate AIC metric based on model type
            if isinstance(model.model, CoxPHFitter):
                metrics['model_performance']['AIC'] = float(model.model.AIC_partial_)
            else:
                metrics['model_performance']['AIC'] = float(model.model.AIC_)
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def _format_metrics(self, metrics):
        """Format metrics for readable terminal output"""
        try:
            output = "\n=== MODEL COMPARISON RESULTS ===\n"
            
            # Extract and sort models by concordance index
            model_performances = []
            for model_name, model_metrics in metrics['model_comparison'].items():
                c_index = model_metrics['model_performance']['concordance_index']
                aic = model_metrics['model_performance']['AIC']
                ll = model_metrics['model_performance']['log_likelihood']
                model_performances.append({
                    'model': model_name,
                    'c_index': c_index,
                    'AIC': aic,
                    'log_likelihood': ll
                })
            
            # Sort by concordance index (higher is better)
            model_performances.sort(key=lambda x: x['c_index'], reverse=True)
            
            # Add ranking summary
            output += "\nMODEL RANKING (by concordance index):\n"
            output += "-" * 80 + "\n"
            output += f"{'Rank':<6}{'Model':<12}{'C-index':<12}{'AIC':<15}{'Log-likelihood':<15}\n"
            output += "-" * 80 + "\n"
            
            for i, perf in enumerate(model_performances, 1):
                output += f"{i:<6}{perf['model']:<12}{perf['c_index']:.4f}     {perf['AIC']:.2f}     {perf['log_likelihood']:.2f}\n"
            
            # Add detailed survival probabilities for best model
            best_model = model_performances[0]['model']
            output += f"\nDETAILED PREDICTIONS FOR BEST MODEL ({best_model}):\n"
            output += "-" * 80 + "\n"
            output += f"{'Time Interval':<15}{'Survival Probability':<20}{'Time Point':<15}\n"
            output += "-" * 80 + "\n"
            
            for interval, values in metrics['model_comparison'][best_model].items():
                if interval != 'model_performance':
                    prob = values['baseline_survival_prob']
                    time = values['actual_time_point']
                    output += f"{interval:<15}{prob:.4f}              {time:.1f} minutes\n"
            
            output += "\nINTERPRETATION:\n"
            output += "- C-index: Values closer to 1 indicate better prediction (0.5 is random)\n"
            output += "- AIC: Lower values indicate better model fit\n"
            output += "- Log-likelihood: Higher values indicate better model fit\n"
            output += "- Survival Probability: Probability of NO train in that interval\n"
            
            return output
            
        except Exception as e:
            raise CustomException(e, sys) from e

def main():
    try:
        # Initialize components
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()
        model_trainer = ModelTrainer()

        # Get train and test paths
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Get transformed data
        train_features, test_features, _ = data_transformation.initiate_data_transformation(
            train_path=train_path,
            test_path=test_path
        )
        
        # Train model and get metrics
        metrics = model_trainer.initiate_model_training(
            train_data=train_features,
            test_data=test_features
        )
        
        logging.info("Model training completed successfully")
        logging.info(f"Model performance metrics: {metrics}")
        
        return metrics
        
    except Exception as e:
        logging.error("Error in model training pipeline")
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    metrics = main()
    print("Final metrics:", metrics)