# Project description

This project aims to predict the likelihood of a train crossing at a railroad crossing in the next 60 minutes. If a user is traveling to a destination and the app predicts a train will block their path, they can be alerted and choose a different route. This saves them time and reduces the vehicle emissions associated with idling.

# Data

- The data for this project is a CSV file containing train traveling schedules over time.
- Trains cross the crossing 8-12 times per day.
- They cross more during daylight hours than at night.
- Each train will block traffic for 4-8 minutes.
- The direction of the train is not always the same, but it doesn't really matter because the crossing gets blocked either way.
- The train schedules are spaced out such that if one train goes by, another is not likely to go for a period of time. That might be as little as 20 minutes or as long as several hours. It can vary, but it's never likely one train will be followed immediately by another.
- There is only one track and one crossing, so trains are never going at the same time in opposite directions.
- Because the train schedules are spaced out, the probability of a train crossing in the future increases as the time increases.
- Users are most interested in the next 60 minutes. What happens hours or days from now is not relevant.
- Users will want the probability broken down by 10-minute intervals. This is because if a train just crossed, the probability is near 0 for the next 10 minutes and the user would likely leave on their trip to avoid the train. However based on the train schedule, the probability will increase over the next 10 minutes and the user will want to know the probability of a train crossing in the next 10 minutes, then the next 10 minutes, and so on.