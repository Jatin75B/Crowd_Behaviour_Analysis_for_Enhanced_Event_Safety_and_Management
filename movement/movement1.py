import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture(r'2.mov')

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Create some random colors
colors = np.random.randint(0, 255, (100, 3))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if 'old_frame' not in locals():
        old_frame = frame_gray
        p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)

    # Calculate optical flow
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        old_frame, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[status == 1]
    good_old = p0[status == 1]

    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        cv2.line(frame, (a, b), (c, d), colors[i].tolist(), 2)
        cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update old frame and points
    old_frame = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
