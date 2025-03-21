import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image, ImageDraw
import numpy as np

# Define class names for classification
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

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Faster R-CNN model for object detection
detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
detector.to(device)
detector.eval()

# Load EfficientNet-B0 model for classification
classifier = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_ftrs = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # Adjust output layer
classifier.load_state_dict(torch.load("C:\\Users\\Aiden\\Project_Coral\\Shark_AI_Detection_Model_V2", map_location=device))
classifier.to(device)
classifier.eval()

# Image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_and_classify(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # Detect objects
        with torch.no_grad():
            detections = detector(image_tensor)[0]

        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        threshold = 0.6  # Confidence threshold for detection

        detected_sharks = []
        draw = ImageDraw.Draw(image)

        for i in range(len(boxes)):
            if scores[i] > threshold:
                x1, y1, x2, y2 = map(int, boxes[i])

                # Crop detected region and classify
                cropped_shark = image.crop((x1, y1, x2, y2))
                cropped_tensor = transform(cropped_shark).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = classifier(cropped_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, predicted_class = torch.max(probabilities, dim=0)

                species = class_names[predicted_class.item()]
                detected_sharks.append({
                    "species": species,
                    "confidence": round(confidence.item(), 4),
                    "bbox": [x1, y1, x2, y2]
                })

                # Draw bounding box & label
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 10), f"{species} ({round(confidence.item() * 100, 1)}%)", fill="red")

        # Save processed image
        output_path = "output.jpg"
        image.save(output_path)

        return {"detections": detected_sharks, "output_image": output_path}

    except Exception as e:
        return {"error": str(e)}
