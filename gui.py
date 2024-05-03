import cv2
import tkinter as tk
from tkinter import Scale
from PIL import Image, ImageTk
import threading

class VideoApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        
        self.slider_value = 50

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.scale = Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, label="Variable Slider", command=self.update_slider)
        self.scale.pack()

        self.update_slider()

        self.delay = 10
        self.update()

        self.window.mainloop()

    def update_slider(self, value=None):
        if value:
            self.slider_value = int(value)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            
            # Do some processing based on the slider value
            processed_frame = self.process_frame(frame)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(processed_frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def process_frame(self, frame):
        # Dummy processing, just a simple brightness adjustment based on the slider value
        alpha = self.slider_value / 50.0  # Scale to 0-2 range
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

VideoApp(tk.Tk(), "Video Processing App")
