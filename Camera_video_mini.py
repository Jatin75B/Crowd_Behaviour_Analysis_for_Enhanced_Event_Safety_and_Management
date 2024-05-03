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

def density(frame):
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

    overlay = cv2.applyColorMap(density_map_resized, cv2.COLORMAP_JET)
    
        # Threshold the density map to identify regions with high density
    threshold = 150  # Adjust this threshold as needed
    _, binary_map = cv2.threshold(density_map_resized, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary map
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the number of people as the number of contours
    num_people = len(contours)
    
    # Resize overlay to match frame dimensions
    #return cv2.putText(cv2.resize(overlay, (frame.shape[1], frame.shape[0])), f'People: {num_people}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    return cv2.resize(overlay, (frame.shape[1], frame.shape[0])), num_people
