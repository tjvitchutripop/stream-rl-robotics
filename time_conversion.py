import datetime
import time

# Get the current timestamp
timestamp = 1756242739

# Convert the timestamp to a datetime object
datetime_object = datetime.datetime.fromtimestamp(timestamp)

# Format the datetime object into a human-readable string
# Example format: YYYY-MM-DD HH:MM:SS
interpretable_time = datetime_object.strftime("%Y-%m-%d %H:%M:%S")

print(f"Raw timestamp: {timestamp}")
print(f"Interpretable time: {interpretable_time}")
