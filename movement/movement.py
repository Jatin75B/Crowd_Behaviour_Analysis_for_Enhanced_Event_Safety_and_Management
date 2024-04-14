import cv2
import numpy as np

# Read the video
cap = cv2.VideoCapture('qwe2.mp4')

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Parameters for optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Feature parameters
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Find the first non-blank frame
while True:
    ret, frame = cap.read()
    if not ret or cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 0:
        break

# Convert the first non-blank frame to grayscale
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Find corners in the first frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]


    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
