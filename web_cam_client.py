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
from object_tracker import ObjectTracker
import argparse
from barcode_reader import BarcodeReader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize extractors globally
text_extractor = ImageTextExtractor()
image_analyzer = ImageAnalyzer()
barcode_reader = BarcodeReader()

class DetectedObject:
    def __init__(self, object_id: int, bbox: list, mask: str, label: str):
        self.id = object_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.mask = mask
        self.base_label = label  # Original label from Grounded-SAM-2
        self.annotation = ""     # Additional description/annotation
        self.barcode = ""        # Detected barcode
        self.confidence = 0.0    # Detection confidence
        self.gpt_vision_pending = False  # New flag to track pending GPT Vision requests

    def get_display_label(self) -> str:
        """Compose display label from object attributes"""
        parts = []        
        parts.append(
            f"SAM2: {self.base_label}{self.id}"
        )
        if self.annotation:
            parts.append(
                f"\nVLLM: {self.annotation}"
            )
        if self.barcode:
            parts.append(
                f"\nBar: {self.barcode}"
            )
        return " ".join(parts)

def process_image(image: np.ndarray, text_prompt: str = "product.", server_url: str = "http://localhost:8765", 
                 use_gpt_vision: bool = False, use_ocr: bool = True, use_tracking: bool = True) -> dict:
    """
    Send image to server for processing and return results with text recognition
    
    Args:
        image: numpy array of the image
        text_prompt: Text prompt for object detection
        server_url: URL of the server
        use_gpt_vision: Whether to use GPT Vision instead of OCR
        use_ocr: Whether to use OCR (ignored if use_gpt_vision is True)
        use_tracking: Whether to use object tracking
    
    Returns:
        dict: Server response containing annotations
    """
    # Initialize/get objects dictionary if not exists
    if not hasattr(process_image, 'objects'):
        process_image.objects = {}

    # Convert numpy array to bytes
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image")
    
    # Prepare the files and data
    files = {
        'file': ('image.jpg', encoded_image.tobytes(), 'image/jpeg')
    }
    
    # Perform the SAM2 request
    try:
        response = requests.post(
            f"{server_url}/process-image/",
            files=files,
            data={'text_prompt': text_prompt}
        )
        response.raise_for_status()
        results = response.json()
        
        # Apply object tracking
        results['annotations'] = process_image.tracker.update(results['annotations'])
        
        # Collect all current object IDs first
        current_object_ids = set()
        if use_gpt_vision:
            for ann in results['annotations']:
                if '[' in ann['label']:
                    try:
                        object_id = int(ann['label'].split('[')[1].split(']')[0])
                        current_object_ids.add(object_id)
                    except (IndexError, ValueError):
                        pass

        # Process each annotation
        for ann in results['annotations']:
            # Get or create object ID
            object_id = None
            if '[' in ann['label']:
                try:
                    object_id = int(ann['label'].split('[')[1].split(']')[0])
                except (IndexError, ValueError):
                    pass
            
            if object_id is not None:
                # Create or update DetectedObject
                if object_id not in process_image.objects:
                    process_image.objects[object_id] = DetectedObject(
                        object_id=object_id,
                        bbox=ann['bbox'],
                        mask=ann['mask'],
                        label=ann['label']
                    )
                # Update existing object
                obj = process_image.objects[object_id]
                obj.bbox = ann['bbox']
                obj.mask = ann['mask']
                obj.confidence = ann.get('confidence', 0.0)

                # Handle GPT Vision processing
                if use_gpt_vision and obj.base_label != "barcode":
                    if not obj.annotation and not obj.gpt_vision_pending:
                        # No annotation and no pending request - start new request
                        x1, y1, x2, y2 = map(int, obj.bbox)
                        cropped_obj = image[y1:y2, x1:x2]
                        
                        success, encoded_obj = cv2.imencode('.jpg', cropped_obj)
                        if success:
                            result = image_analyzer.describe_image(
                                encoded_obj.tobytes(),
                                object_id=object_id
                            )
                            obj.gpt_vision_pending = True
                            obj.annotation = "Processing..."
                        else:
                            obj.annotation = 'Image encoding failed'
                    
                    elif obj.gpt_vision_pending:
                        # Check for result if request is pending
                        result = image_analyzer.get_pending_result(object_id)
                        if result and result["status"] == "completed":
                            obj.gpt_vision_pending = False
                            if "error" in result:
                                obj.annotation = "Error: " + result["error"]
                            else:
                                obj.annotation = result.get('description', '...')

                # Only process barcode if we haven't found one yet
                if not obj.barcode:
                    x1, y1, x2, y2 = map(int, obj.bbox)
                    for other_ann in results['annotations']:
                        if 'barcode' in other_ann['label'].lower():
                            bx1, by1, bx2, by2 = map(int, other_ann['bbox'])
                            
                            # Check if barcode bbox is within current object bbox
                            if (bx1 >= x1 and bx2 <= x2 and by1 >= y1 and by2 <= y2):
                                # Extract barcode region
                                barcode_region = image[by1:by2, bx1:bx2]
                                
                                try:
                                    barcode_results = barcode_reader.read_barcode(barcode_region)
                                    if barcode_results:
                                        obj.barcode = barcode_results[0].data
                                        break
                                except Exception as e:
                                    logger.warning(f"Failed to read barcode: {e}")

        # Clean up old objects after processing all annotations
        if use_gpt_vision:
            image_analyzer.cleanup_old_objects(current_object_ids)
        
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
        
        # Draw mask and bounding box
        mask = mask_util.decode(ann['mask'])
        mask_overlay = vis_image.copy()
        mask_overlay[mask > 0] = np.array(color) * 0.5 + mask_overlay[mask > 0] * 0.5
        vis_image = mask_overlay
        
        bbox = ann['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Get object ID if present
        object_id = None
        if '[' in ann['label']:
            try:
                object_id = int(ann['label'].split('[')[1].split(']')[0])
            except (IndexError, ValueError):
                pass
        
        # Get display text from DetectedObject if available
        if hasattr(process_image, 'objects') and object_id is not None and object_id in process_image.objects:
            text = process_image.objects[object_id].get_display_label()
            
            # Draw text with background
            lines = text.split('\n')
            for i, line in enumerate(lines):
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_y = y1 - 10 - (text_height + 5) * (len(lines) - 1 - i)  # Stack text lines from bottom up
                cv2.rectangle(vis_image, (x1, text_y - text_height), (x1 + text_width, text_y + 5), 
                            color, -1)
                cv2.putText(vis_image, line, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            # Fallback to original text display if object not tracked
            label_confidence = f"{ann['label']}: {ann['confidence']:.2f}"
            texts = [label_confidence]
            if 'extracted_text' in ann:
                extracted_text = f"Description: {ann['extracted_text']}"
                texts.append(extracted_text)
            
            # Draw text with background (original method)
            for i, text in enumerate(texts):
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_y = y1 - 10 - (text_height + 5) * (1 - i)
                cv2.rectangle(vis_image, (x1, text_y - text_height), (x1 + text_width, text_y + 5), 
                            color, -1)
                cv2.putText(vis_image, text, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

def main():
    # Get API key and other settings from command line arguments or config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', help='OpenAI API key')
    args = parser.parse_args()

    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading config.json: {e}")
        return

    if config.get('use_gpt_vision', False) == True:
        # Get API key from args or config
        api_key = args.api_key or config.get('api_key')
        if not api_key:
            logger.error("API key not provided and couldn't be loaded from config.json")
            return
    else:
        api_key = None

    # Get processing parameters from config
    use_ocr = config.get('use_ocr', False)
    use_gpt_vision = config.get('use_gpt_vision', False)

    # Initialize tracker (now always initialized)
    tracking_distance_threshold = config.get('tracking_distance_threshold', 100.0)
    process_image.tracker = ObjectTracker(
        max_frames=5, 
        distance_threshold=tracking_distance_threshold
    )
    logger.info(f"Initialized object tracker with distance threshold: {tracking_distance_threshold}")

    # Initialize image analyzer with API key
    global image_analyzer
    image_analyzer = ImageAnalyzer(api_key=api_key)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    # Get resolution settings from config
    camera_config = config.get('camera', {})
    resolutions = camera_config.get('resolutions', {})
    default_resolution = camera_config.get('default_resolution', '720p')
    
    # Get resolution values
    if default_resolution not in resolutions:
        logger.warning(f"Invalid resolution {default_resolution}, defaulting to 720p")
        width, height = (1280, 720)  # 720p fallback
    else:
        width, height = resolutions[default_resolution]
        
    logger.info(f"Setting resolution to {default_resolution} ({width}x{height})")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Actual resolution: {actual_width}x{actual_height}")
    
    text_prompt = "product. barcode." # The point after each prompt is crucial.
    
    # Add frame counter before the while loop
    frame_counter = 0
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

        # Increment frame counter
        frame_counter += 1

        # Process frame
        results = process_image(frame, text_prompt, 
                              use_gpt_vision=use_gpt_vision, 
                              use_ocr=use_ocr,
                              use_tracking=True)
        
        # After running the model and getting results
        if results is not None and len(results['annotations']) > 0:  # Check if we have any detections
            if not objects_saved and frame_counter > 10:  # Only save after 10th frame
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
                    
                    # If the detected label contains "barcode", skip mask and save only the cropped bounding box
                    if 'barcode' in ann['label'].lower():
                        # Just crop from the original frame
                        cropped_obj = frame[y1:y2, x1:x2]
                        final_obj = cropped_obj
                    else:
                        # Original approach: replace everything outside mask with green
                        
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