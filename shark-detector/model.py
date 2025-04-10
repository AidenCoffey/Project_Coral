import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image, ImageDraw
import numpy as np
import datetime
from PIL import ImageFont

# Define class names for classification
class_names = ['Black tip', 'Bonnethead', 'Nurse shark', 'Sand tiger', 'Spinner', 'Thresher', 'blue', 'bull', 'hammer', 'lemon', 'mako', 'tiger', 'tip', 'whale', 'white']

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
classifier.load_state_dict(torch.load("C:\\Users\\Aiden\\Project_Coral\\shark_data_shart_edition_2.pth", map_location=device))
classifier.to(device)
classifier.eval()

# Image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CONFIDENCE_THRESHOLD = 0.65

def interpolate_color(confidence):
    """Interpolates color from blue (low confidence) to green (mid confidence) to red (high confidence)."""
    if confidence <= 0.7:
        return (0, 0, 255)  # Blue for low confidence
    elif confidence <= 0.8:
        ratio = (confidence - 0.7) / 0.1
        r = int(0 + (0 * ratio))  # Still 0 (blue to green transition)
        g = int(0 + (255 * ratio))  # Increasing green
        b = int(255 - (255 * ratio))  # Decreasing blue
        return (r, g, b)
    else:
        ratio = (confidence - 0.8) / 0.2
        r = int(0 + (255 * ratio))  # Increasing red
        g = int(255 - (255 * ratio))  # Decreasing green
        b = 0  # Fully red transition
        return (r, g, b)

def detect_and_classify(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        target_size = (640, 400)  # Adjust as needed
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert image to tensor
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # Detect objects
        with torch.no_grad():
            detections = detector(image_tensor)[0]

        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        threshold = 0.6  # Confidence threshold for detection in Faster R-CNN

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
                confidence_value = confidence.item()

                # if confidence_value >= CONFIDENCE_THRESHOLD:
                detected_sharks.append({
                    "species": species,
                    "confidence": round(confidence_value, 4),
                    "bbox": [x1, y1, x2, y2]
                })

                # Determine bounding box color
                bbox_color = interpolate_color(confidence_value)

                # Draw bounding box & label
                draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=3)
                draw.text((x1, y1 - 10), f"{species} ({round(confidence_value * 100, 1)}%)", fill=bbox_color)

        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        font = ImageFont.load_default()  # Default font
        text_size = draw.textbbox((0, 0), timestamp, font=font)
        text_x = image.width - text_size[2] - 10
        text_y = image.height - text_size[3] - 10
        draw.text((text_x, text_y), timestamp, fill="white", font=font)

        # Save processed image
        output_path = "output.jpg"
        image.save(output_path)

        return {"detections": detected_sharks, "output_image": output_path}

    except Exception as e:
        return {"error": str(e)}
