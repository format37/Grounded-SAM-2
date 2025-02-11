import requests
import logging
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_image(image: np.ndarray, text_prompt: str = "car. tyre.", server_url: str = "http://localhost:8765") -> dict:
    """
    Send image to server for processing and return results
    
    Args:
        image: numpy array of the image
        text_prompt: Text prompt for object detection
        server_url: URL of the server
    
    Returns:
        dict: Server response containing annotations
    """
    # Convert numpy array to bytes
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image")
    
    # Prepare the files and data
    files = {
        'file': ('image.jpg', encoded_image.tobytes(), 'image/jpeg')
    }
    
    # Make the request
    try:
        response = requests.post(
            f"{server_url}/process-image/",
            files=files,
            data={'text_prompt': text_prompt}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to server: {e}")
        raise

def visualize_results(image: np.ndarray, results: dict) -> np.ndarray:
    """
    Visualize detection results with bounding boxes, masks, and confidence scores
    
    Args:
        image: numpy array of the original image
        results: Dictionary containing detection results
    Returns:
        numpy array of the visualized image
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Generate random colors for each instance
    colors = np.random.randint(0, 255, size=(len(results['annotations']), 3))
    
    # Draw each detection
    for idx, ann in enumerate(results['annotations']):
        color = colors[idx].tolist()
        
        # Draw mask
        mask = mask_util.decode(ann['mask'])
        mask_overlay = vis_image.copy()
        mask_overlay[mask > 0] = np.array(color) * 0.5 + mask_overlay[mask > 0] * 0.5
        vis_image = mask_overlay
        
        # Draw bounding box
        bbox = ann['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label and confidence
        label = f"{ann['label']}: {ann['confidence']:.2f}"
        cv2.putText(vis_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to BGR for display
    return cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    text_prompt = "eye. nose." # The point after each prompt is crucial.
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame from webcam")
                break

            # Process frame
            results = process_image(frame, text_prompt)
            
            # Visualize results
            vis_frame = visualize_results(frame, results)
            
            # Display the result
            cv2.imshow('Object Detection', vis_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.error(f"Error processing webcam feed: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
