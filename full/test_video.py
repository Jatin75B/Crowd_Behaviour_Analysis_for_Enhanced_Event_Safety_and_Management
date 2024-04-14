import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the model
md1_path = r'D:\Jatin\College stuff\Major Project\Crowd\crowd-density-aspp\Models\aspp\aspp_1_800.pt'
md1 = torch.load(md1_path, map_location=device).to(device)
md1.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Read video file
video_path = r'D:\Jatin\College stuff\Major Project\Crowd\crowd-density-aspp\263C044_060_c.mov'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set resize parameters
resize_width = 640  # Change this according to your preference
resize_height = 480  # Change this according to your preference

# Create window to display analyzed frames
cv2.namedWindow('Analyzed Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Analyzed Frame', resize_width, resize_height)

# Process each frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))

    # Convert frame to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformation
    input_image = transform(frame_pil).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output_density_map = md1(input_image)

    # Convert the output density map tensor to numpy array
    density_map_numpy = output_density_map.squeeze().cpu().numpy()

    # Resize density map to match frame size
    density_map_resized = cv2.resize(density_map_numpy, (resize_width, resize_height))

    # Normalize density map for visualization
    density_map_resized = cv2.normalize(density_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Overlay density map on frame
    alpha = 0.5
    overlay = cv2.applyColorMap(density_map_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1, overlay, alpha, 0)

    combined_image_vertical = cv2.vconcat([frame, overlay])
    
    # Display the analyzed frame
    cv2.imshow('Analyzed Frame', combined_image_vertical)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display progress
    frame_count += 1
    print(f'Processed frame {frame_count}/{total_frames}')

# Release resources
cap.release()
cv2.destroyAllWindows()
