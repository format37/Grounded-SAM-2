import requests
import logging
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
import time
import os
from local_ocr import ImageTextExtractor
from gpt_vision import ImageAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize extractors globally
text_extractor = ImageTextExtractor()
image_analyzer = ImageAnalyzer()

def process_image(image: np.ndarray, text_prompt: str = "car. tyre.", server_url: str = "http://localhost:8765", use_gpt_vision: bool = False, use_ocr: bool = True) -> dict:
    """
    Send image to server for processing and return results with text recognition
    
    Args:
        image: numpy array of the image
        text_prompt: Text prompt for object detection
        server_url: URL of the server
        use_gpt_vision: Whether to use GPT Vision instead of OCR
        use_ocr: Whether to use OCR (ignored if use_gpt_vision is True)
    
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
        results = response.json()
        
        # Extract text from each detected object
        for ann in results['annotations']:
            # Decode mask
            mask = mask_util.decode(ann['mask'])
            
            # Create masked image
            masked_obj = np.zeros_like(image)
            masked_obj[mask > 0] = image[mask > 0]
            
            # Get bbox coordinates and crop
            x1, y1, x2, y2 = map(int, ann['bbox'])
            cropped_obj = masked_obj[y1:y2, x1:x2]
            
            if use_gpt_vision:
                # Convert numpy array to bytes
                success, encoded_obj = cv2.imencode('.jpg', cropped_obj)
                if success:
                    # Get description from GPT Vision
                    description = image_analyzer.describe_image(encoded_obj.tobytes())
                    ann['extracted_text'] = description.get('description', '...')
                else:
                    ann['extracted_text'] = '...'
            elif use_ocr:
                # Use OCR method
                extracted_text = text_extractor.extract_text(cropped_obj)
                ann['extracted_text'] = extracted_text if extracted_text else "..."
            else:
                # No text extraction
                ann['extracted_text'] = "..."
        
        return results
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to server: {e}")
        raise

def visualize_results(image: np.ndarray, results: dict) -> np.ndarray:
    """
    Visualize detection results with bounding boxes, masks, confidence scores, and extracted text
    
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
        
        # Update label to include extracted text
        label = f"{ann['label']} ({ann['extracted_text']}): {ann['confidence']:.2f}"
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
    
    # Set resolution to 2048x1536
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Actual resolution: {actual_width}x{actual_height}")

    # text_prompt = ''
    # text_prompt+='household/paper_goods.'
    # text_prompt+=' beverages/tea.'
    # text_prompt+=' dairy/cream.'
    # text_prompt+=' groceries/sweeteners.'
    # text_prompt+=' toys/puzzles.'
    # text_prompt+=' groceries/seasonings.'

    # text_prompt = ''
    # text_prompt+='toilet_paper.'
    # text_prompt+=' tea.'
    # text_prompt+=' cream.'
    # text_prompt+=' salt.'
    # text_prompt+=' toy.'
    # text_prompt+=' sweetener.'
    
    text_prompt = "product." # The point after each prompt is crucial.
    
    use_ocr = False
    use_gpt_vision = False
    
    
    objects_saved = False
    
    # Add FPS calculation variables
    prev_frame_time = 0
    curr_frame_time = 0
    
    while True:
        # Calculate FPS
        curr_frame_time = time.time()
        fps = 1/(curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = curr_frame_time
        
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame from webcam")
            break

        # Process frame
        results = process_image(frame, text_prompt, use_gpt_vision=use_gpt_vision, use_ocr=use_ocr)
        
        # After running the model and getting results
        if results is not None and len(results['annotations']) > 0:  # Check if we have any detections
            if not objects_saved:
                # Log original image dimensions
                height, width = frame.shape[:2]
                logger.info(f"Original image dimensions: {width}x{height}")
                # Create directory if it doesn't exist
                save_dir = "./notebooks/images/objects"
                os.makedirs(save_dir, exist_ok=True)
                
                # Process each detected object
                for i, ann in enumerate(results['annotations']):
                    # Get bbox coordinates
                    x1, y1, x2, y2 = map(int, ann['bbox'])
                    
                    # Create a blank green box image
                    obj_height = y2 - y1
                    obj_width = x2 - x1
                    green_box = np.zeros((obj_height, obj_width, 3), dtype=np.uint8)
                    green_box[:, :] = [0, 255, 0]  # Green color
                    
                    # Create mask for this object
                    mask = mask_util.decode(ann['mask'])
                    
                    # Extract object from original image using mask
                    masked_obj = np.zeros_like(frame)
                    masked_obj[mask > 0] = frame[mask > 0]
                    
                    # Crop the masked object to bbox
                    cropped_obj = masked_obj[y1:y2, x1:x2]
                    
                    # Combine with green background
                    final_obj = np.where(cropped_obj != 0, cropped_obj, green_box)
                    
                    # Save the image
                    filename = f"image_{i}_{ann['label']}_{ann['confidence']:.2f}.png"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, final_obj)
                
                objects_saved = True  # Mark that we've saved the objects
                logger.info(f"Saved {len(results['annotations'])} objects to {save_dir}")
        
        # Visualize results
        vis_frame = visualize_results(frame, results)
        
        # Add FPS text to the frame
        cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow('Object Detection', vis_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
