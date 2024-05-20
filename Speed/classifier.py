import numpy as np

RUNNING_THRESHOLD = 4
FASTER_MOVEMENT_THRESHOLD= 7


def classify_movement1(speed):
    # Set a threshold to classify movement as 'Running' or 'Not Running'
    threshold = 5.0
    if speed > threshold:
        return 'Running'
    else:
        return 'Not Running'

def classify_movement(speed):

    if np.isnan(speed):
        return 'Unknown'  # Handle NaN values

    if speed >= FASTER_MOVEMENT_THRESHOLD:
        return 'Faster Movement'
    elif speed >=RUNNING_THRESHOLD:
        return 'Running'
    else:
        return 'Not Running'

