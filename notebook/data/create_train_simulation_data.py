import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_train_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize empty list to store records
    records = []
    
    # Define date range
    start_date = datetime(2024, 10, 20)
    end_date = datetime(2024, 11, 20)
    current_date = start_date
    
    # Longitude and latitude ranges (NYC area)
    long_range = (-74.042, -73.578)
    lat_range = (40.7107, 40.7156)
    
    train_id = 1
    
    while current_date <= end_date:
        # Random number of trains for this day (8-12)
        num_trains = random.randint(8, 12)
        
        # Generate daytime hours (weighted towards 7am-7pm)
        hours = np.random.choice(
            np.concatenate([
                np.repeat(np.arange(7, 19), 3),  # Higher weight for 7am-7pm
                np.arange(19, 22)                # Lower weight for evening
            ]),
            size=num_trains,
            replace=False
        )
        hours.sort()
        
        # For each train in this day
        for hour in hours:
            # Generate random minute
            minute = random.randint(0, 59)
            
            # Create time_start
            time_start = current_date.replace(hour=hour, minute=minute)
            
            # Random passing duration (5-8 minutes)
            passing_duration = random.randint(5, 8)
            
            # Calculate time_end
            time_end = time_start + timedelta(minutes=passing_duration)
            
            # Alternate directions (roughly)
            direction = 'E' if train_id % 2 == 0 else 'W'
            
            # Random speed between 30-50
            speed = random.randint(30, 50)
            
            # Random coordinates within range
            longitude = random.uniform(long_range[0], long_range[1])
            latitude = random.uniform(lat_range[0], lat_range[1])
            
            # Create record
            record = {
                'train_id': f'TR{str(train_id).zfill(3)}',
                'time_start': time_start.strftime('%Y-%m-%d %H:%M:%S'),
                'time_end': time_end.strftime('%Y-%m-%d %H:%M:%S'),
                'direction': direction,
                'speed': speed,
                'longitude': round(longitude, 6),
                'latitude': round(latitude, 6)
            }
            
            records.append(record)
            train_id += 1
        
        current_date += timedelta(days=1)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by time_start
    df = df.sort_values('time_start')
    
    return df

# Generate the data
df = generate_train_data()

# Save to CSV
df.to_csv('notebook/data/train_positions.csv', index=False)

# Print summary statistics
print(f"Total records generated: {len(df)}")
print("\nRecords per day:")
df['date'] = pd.to_datetime(df['time_start']).dt.date
print(df.groupby('date').size().describe())