import torch
import torchvision.transforms as transforms
from PIL import Image
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

# Camera index (usually 0 for the default camera)
camera_index = 0

# Open camera capture
cap = cv2.VideoCapture(camera_index)

# Set capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create window to display analyzed frames
cv2.namedWindow('Analyzed Frame', cv2.WINDOW_NORMAL)

# Process each frame from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

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
    density_map_resized = cv2.resize(density_map_numpy, (frame.shape[1], frame.shape[0]))

    # Normalize density map for visualization
    density_map_resized = cv2.normalize(density_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Overlay density map on frame
    alpha = 0.5
    overlay = cv2.applyColorMap(density_map_resized, cv2.COLORMAP_JET)

    # Resize overlay to match frame dimensions
    overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))

    # Combine frame and overlay
    combined = cv2.addWeighted(frame, 1, overlay, alpha, 0)
    
    
    combined_image_vertical = cv2.vconcat([frame, overlay])
    
    # Display the analyzed frame
    cv2.imshow('Analyzed Frame', combined_image_vertical)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
