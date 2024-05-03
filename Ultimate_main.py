import time
import threading
from main24 import twoon
from movement.yomov_copy import oneon
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

def choose_input():
    while True:
        choice = input("Choose input source (camera or video): ").lower()
        if choice == 'camera':
            return 0  # Return camera index
        elif choice == 'video':
            video_path = input("Enter video file path: ")
            return video_path  # Return video file path
        else:
            print("Invalid input. Please choose 'camera' or 'video'.")

input_source = choose_input()

# Create threads for each function with arguments
thread1 = threading.Thread(target=twoon, args=(input_source))
thread2 = threading.Thread(target=oneon, args=(input_source))

# Start both threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both threads have finished executing.")
