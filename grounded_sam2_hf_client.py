import requests
import logging
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_image(image_path: str, text_prompt: str = "car. tire.", server_url: str = "http://localhost:8000") -> dict:
    """
    Send image to server for processing and return results
    
    Args:
        image_path: Path to the image file
        text_prompt: Text prompt for object detection
        server_url: URL of the server
    
    Returns:
        dict: Server response containing annotations
    """
    # Prepare the files and data
    files = {
        'file': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    # Make the request
    try:
        response = requests.post(
            f"{server_url}/process-image/",
            files=files,
            params={'text_prompt': text_prompt}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to server: {e}")
        raise

def visualize_results(image_path: str, results: dict, output_path: str = "output.jpg"):
    """
    Visualize detection results with bounding boxes, masks, and confidence scores
    
    Args:
        image_path: Path to the original image
        results: Dictionary containing detection results
        output_path: Path to save the visualization
    """
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Generate random colors for each instance
    colors = np.random.randint(0, 255, size=(len(results['annotations']), 3))
    
    # Draw each detection
    for idx, ann in enumerate(results['annotations']):
        # Get the color for this instance
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
    
    # Convert back to BGR for saving
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, vis_image)
    logger.info(f"Visualization saved to {output_path}")

def main():
    # Define parameters
    image_path = "notebooks/images/truck.jpg"
    text_prompt = "car. tire."
    
    # Check if image exists
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    logger.info(f"Processing image: {image_path}")
    logger.info(f"Text prompt: {text_prompt}")
    
    try:
        # Send request to server
        results = process_image(image_path, text_prompt)
        
        # Log results
        logger.info("Received response from server:")
        logger.info(f"Number of detected objects: {len(results['annotations'])}")
        
        # Visualize results
        visualize_results(image_path, results)
        
        # Print detailed results
        print("\nDetailed results:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
