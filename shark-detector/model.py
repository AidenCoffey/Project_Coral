import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define class names
class_names = [
    "Blue Shark", 
    "Whale Shark",   
    "Nurse Shark",
    "Tiger Shark",
    "White Tip Shark",
    "Sand Tiger Shark",
    "Bull Shark",
    "Hammerhead Shark",    
    "White Shark",     
    "Mako Shark"
]


# Load the EfficientNet-B0 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # Adjust output layer

# Load trained model weights
model_path = 'C:\\Users\\Aiden\\Project_Coral\\Shark_AI_Detection_Model_V2'
model.load_state_dict(torch.load(model_path, map_location=device))  # Use map_location
model.to(device)
model.eval()

# Define preprocessing (should match training)
transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data_transforms_test = transforms.Compose([
    transforms.Resize(320),             # Resize to 224x224
    transforms.CenterCrop(320),         # Center crop
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

def predict_shark(image_path):
    try: 
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_transformed = data_transforms_test(image).unsqueeze(0)  # Add batch dimension

        # Move the image to the device (GPU or CPU)
        image_transformed = image_transformed.to(device)

        # Make prediction
        with torch.no_grad():  # No gradient calculation needed for inference
            output = model(image_transformed)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Convert output to probabilities
            confidence, predicted_class = torch.max(probabilities, dim=0)  # Get the highest probability and class index

            result = {
                "detected": True,
                "species": class_names[predicted_class.item()],
                "confidence": round(confidence.item(), 4)
            }
            return result

    except Exception as e:
        return {"error": str(e)}
