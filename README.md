# Train Crossing Prediction Model

This project predicts the likelihood of a train crossing in the next 60 minutes using survival analysis models.

## Overview
- Uses historical train crossing data to predict future crossings
- Compares different survival models (Cox, Weibull, LogNormal, LogLogistic)
- Provides probabilities for 10-minute intervals up to 60 minutes ahead

## Model Performance
- Best model: LogNormal (C-index: 0.647)
- Provides survival probabilities for each 10-minute interval
- Accounts for time-of-day and historical patterns

## Project Structure