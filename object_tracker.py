#!/usr/bin/env python3
import json
import os
import logging
from typing import List, Dict, Any
import cv2
import numpy as np
import requests
import pycocotools.mask as mask_util
from barcode_reader import BarcodeReader
from datetime import datetime
from webcam_handler import WebcamHandler

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


class ObjectTracker:
    """
    Class to handle object detection, tracking, and maintaining a limited history of frames.
    """
    
    def __init__(self, config_path: str, server_url: str = "http://localhost:8765"):
        """
        Initialize the tracker by loading configuration and setting up internal states.
        :param config_path: Path to a configuration JSON file.
        :param server_url: URL of the SAM2 server.
        """
        self.config = self.load_config(config_path)
        self.max_frames = self.config.get("max_frames", 10)
        self.tracking_threshold_h = self.config.get("tracking_threshold_h", 5.0)  # percentage of image height
        self.tracking_threshold_w = self.config.get("tracking_threshold_w", 5.0)  # percentage of image width
        self.frames: List[Dict[int, Dict[str, Any]]] = []
        self.next_object_id = 0
        self.server_url = server_url
        self.image_height = None
        self.image_width = None
        self.barcode_reader = BarcodeReader()  # Initialize barcode reader
        self.object_barcodes = {}  # Store barcodes by object ID
        self.object_descriptions = {}  # Store descriptions by object ID
        self.description_pending = set()  # Track which objects have pending descriptions
        self.image_analyzer = None  # Will be initialized when needed
        logging.info(f"ObjectTracker initialized with max_frames={self.max_frames}")

    def load_config(self, config_path: str) -> dict:
        """
        Load configuration from a JSON file.
        :param config_path: Path to the configuration file.
        :return: Dictionary with configuration parameters.
        """
        if not os.path.exists(config_path):
            logging.error(f"Config file '{config_path}' not found. Using default configuration.")
            return {}
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                logging.info(f"Configuration loaded: {config}")
                return config
        except Exception as e:
            logging.error(f"Error reading config file: {e}")
            return {}

    def init_image_analyzer(self):
        """Lazy initialization of image analyzer"""
        if self.image_analyzer is None:
            from gpt_vision import ImageAnalyzer
            api_key = self.config.get('api_key')
            if not api_key:
                logging.error("No API key found in config for GPT Vision")
                return False
            self.image_analyzer = ImageAnalyzer(api_key=api_key)
            return True
        return True

    def process_frame(self, image: np.ndarray, text_prompt: str = "product.") -> None:
        """
        Process a new frame: detect objects, perform tracking, and update frames history.
        :param image: The image frame as a NumPy array.
        :param text_prompt: Text prompt for object detection
        """
        self.image_height, self.image_width = image.shape[:2]
        logging.debug("Processing new frame...")
        detected_objects = self.detect_objects(image, text_prompt)
        tracked_objects = self.track_objects(detected_objects)
        
        # Process barcodes and descriptions for tracked objects
        self.process_barcodes(image, tracked_objects, detected_objects)
        if self.config.get('use_gpt_vision', False):
            self.process_descriptions(image, tracked_objects)
        
        # Add timestamp to frame data
        frame_data = {
            'timestamp': datetime.now(),
            'objects': tracked_objects
        }
        
        self.frames.append(frame_data)
        self.limit_frames_history()
        logging.info(f"Frame processed.")

    def detect_objects(self, image: np.ndarray, text_prompt: str) -> List[Dict[str, Any]]:
        """
        Real object detection using SAM2 server
        :param image: The image frame as a NumPy array
        :param text_prompt: Text prompt for object detection
        :return: List of detected object dictionaries
        """
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
                f"{self.server_url}/process-image/",
                files=files,
                data={'text_prompt': text_prompt}
            )
            response.raise_for_status()
            results = response.json()
            
            # Return the annotations list which contains the detected objects
            return results['annotations']
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error making request to server: {e}")
            return []

    def is_bbox_similar(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """
        Compare two bounding boxes using configurable thresholds based on image dimensions.
        :param bbox1: First bounding box [x1, y1, x2, y2].
        :param bbox2: Second bounding box [x1, y1, x2, y2].
        :return: True if bboxes are considered similar, False otherwise.
        """
        if self.image_height is None or self.image_width is None:
            logging.error("Image dimensions not set. Cannot compare bounding boxes.")
            return False

        threshold_w = (self.tracking_threshold_w / 100.0) * self.image_width
        threshold_h = (self.tracking_threshold_h / 100.0) * self.image_height

        # Calculate center points of both bounding boxes
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2

        # Calculate distances between centers
        dx = abs(center1_x - center2_x)
        dy = abs(center1_y - center2_y)

        # Calculate IoU (Intersection over Union)
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            iou = 0.0
        else:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            iou = intersection / (area1 + area2 - intersection)

        similar = (dx < threshold_w and dy < threshold_h) or iou > 0.5
        logging.debug(f"Comparing bboxes: dx={dx:.2f}, dy={dy:.2f}, IoU={iou:.2f} -> similar: {similar}")
        return similar

    def track_objects(self, detected_objects: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Track objects across frames using configurable thresholds and IoU.
        :param detected_objects: List of objects detected in the current frame.
        :return: A dictionary mapping object_id to object data.
        """
        frame_objects: Dict[int, Dict[str, Any]] = {}
        
        # If this is the first frame, assign new IDs to all objects
        if not self.frames:
            for obj in detected_objects:
                frame_objects[self.next_object_id] = obj
                self.next_object_id += 1
            return frame_objects

        # Get previous frame's objects
        prev_frame = self.frames[-1]['objects']
        
        # Create lists of unmatched objects
        unmatched_detections = list(range(len(detected_objects)))
        unmatched_trackers = list(prev_frame.keys())
        
        # Match objects based on bbox similarity
        for det_idx in unmatched_detections[:]:
            det_obj = detected_objects[det_idx]
            best_match = None
            best_match_id = None
            
            for tracker_id in unmatched_trackers[:]:
                prev_obj = prev_frame[tracker_id]
                if self.is_bbox_similar(det_obj["bbox"], prev_obj["bbox"]):
                    best_match = tracker_id
                    best_match_id = det_idx
                    break
            
            if best_match is not None:
                frame_objects[best_match] = detected_objects[best_match_id]
                unmatched_detections.remove(best_match_id)
                unmatched_trackers.remove(best_match)
        
        # Assign new IDs to unmatched detections
        for det_idx in unmatched_detections:
            frame_objects[self.next_object_id] = detected_objects[det_idx]
            self.next_object_id += 1

        return frame_objects

    def limit_frames_history(self) -> None:
        """
        Keep the frames history within the maximum count as specified in the configuration.
        """
        if len(self.frames) > self.max_frames:
            removed = len(self.frames) - self.max_frames
            self.frames = self.frames[-self.max_frames:]
            logging.debug(f"Frames history limited. Removed {removed} old frame(s).")

    # Placeholder for future asynchronous processing (e.g., calling VLLM for object description)
    def enqueue_object_description(self, object_id: int, object_data: Dict[str, Any]) -> None:
        """
        Placeholder method to enqueue an object for asynchronous description processing.
        :param object_id: The ID of the object.
        :param object_data: The object data.
        """
        # TODO: Implement asynchronous VLLM object description.
        logging.debug(f"Enqueueing object {object_id} for async description (not implemented yet).")

    def visualize_frame(self, image: np.ndarray, frame_objects: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """
        Visualize detection results with bounding boxes, masks, and annotations
        
        Args:
            image: numpy array of the original image
            frame_objects: Dictionary of tracked objects in the current frame
        Returns:
            numpy array of the visualized image
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image = image.copy()
        
        # Generate random colors for each instance
        colors = np.random.randint(0, 255, size=(len(frame_objects), 3))
        
        # Draw each detection
        for idx, (obj_id, obj_data) in enumerate(frame_objects.items()):
            color = colors[idx].tolist()
            
            # Draw mask and bounding box
            mask = mask_util.decode(obj_data['mask'])
            mask_overlay = vis_image.copy()
            mask_overlay[mask > 0] = np.array(color) * 0.5 + mask_overlay[mask > 0] * 0.5
            vis_image = mask_overlay
            
            # Draw bounding box
            bbox = obj_data['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare display text with description
            confidence = obj_data.get('confidence', 0) * 100
            label = f"({obj_id}) {obj_data['label']} {confidence:.1f}%"
            if 'barcode' in obj_data:
                label += f"\nBar: {obj_data['barcode']}"
            if 'description' in obj_data:
                label += f"\nVLM: {obj_data['description']}"
            
            # Draw multi-line text with background
            lines = label.split('\n')
            for i, line in enumerate(lines):
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_y = y1 - 10 - (text_height + 5) * (len(lines) - 1 - i)
                cv2.rectangle(vis_image, (x1, text_y - text_height), (x1 + text_width, text_y + 5), 
                             color, -1)
                cv2.putText(vis_image, line, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    def save_mask_images(self, image: np.ndarray, frame_objects: Dict[int, Dict[str, Any]], output_dir: str) -> None:
        """
        Save individual mask images for each detected object
        
        Args:
            image: numpy array of the original image
            frame_objects: Dictionary of tracked objects in the current frame
            output_dir: Directory to save the mask images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for obj_id, obj_data in frame_objects.items():
            # Decode the mask
            mask = mask_util.decode(obj_data['mask'])
            
            # Create a mask image (white mask on black background)
            mask_image = np.zeros_like(image)
            mask_image[mask > 0] = [255, 255, 255]
            
            # Create masked original image
            masked_original = image.copy()
            masked_original[mask == 0] = 0
            
            # Save both versions
            mask_filename = os.path.join(output_dir, f'mask_{obj_id}.png')
            masked_orig_filename = os.path.join(output_dir, f'masked_original_{obj_id}.png')
            
            cv2.imwrite(mask_filename, mask_image)
            cv2.imwrite(masked_orig_filename, masked_original)
            
            logging.debug(f"Saved mask images for object {obj_id}: {mask_filename}, {masked_orig_filename}")

    def process_barcodes(self, image: np.ndarray, tracked_objects: Dict[int, Dict[str, Any]], 
                        detected_objects: List[Dict[str, Any]]) -> None:
        """
        Process barcodes for tracked objects.
        :param image: The image frame as a NumPy array
        :param tracked_objects: Dictionary of tracked objects
        :param detected_objects: List of all detected objects including barcodes
        """
        # Find barcode annotations
        barcode_annotations = [
            ann for ann in detected_objects 
            if 'barcode' in ann.get('label', '').lower()
        ]
        
        # Process each tracked object
        for obj_id, obj_data in tracked_objects.items():
            # Check if barcode already exists for this object ID
            if obj_id in self.object_barcodes:
                obj_data['barcode'] = self.object_barcodes[obj_id]
                logging.info(f"Using cached barcode for object {obj_id}: {obj_data['barcode']}")
                continue
            
            # Get object bbox
            obj_x1, obj_y1, obj_x2, obj_y2 = map(int, obj_data['bbox'])
            
            # Check each barcode annotation
            for barcode_ann in barcode_annotations:
                # Get barcode bbox
                bx1, by1, bx2, by2 = map(int, barcode_ann['bbox'])
                
                # Check if barcode bbox is within object bbox
                if (bx1 >= obj_x1 and bx2 <= obj_x2 and 
                    by1 >= obj_y1 and by2 <= obj_y2):
                    
                    # Extract barcode region
                    barcode_region = image[by1:by2, bx1:bx2]
                    
                    # Read barcode
                    barcode_results = self.barcode_reader.read_barcode(barcode_region)
                    if barcode_results:
                        barcode_data = barcode_results[0].data
                        # Store barcode in both the object data and our cache
                        obj_data['barcode'] = barcode_data
                        self.object_barcodes[obj_id] = barcode_data
                        logging.info(f"Found and cached barcode for object {obj_id}: {barcode_data}")
                        break

    def process_descriptions(self, image: np.ndarray, tracked_objects: Dict[int, Dict[str, Any]]) -> None:
        """
        Process VLM descriptions for tracked objects.
        :param image: The image frame as a NumPy array
        :param tracked_objects: Dictionary of tracked objects
        """
        if not self.init_image_analyzer():
            return

        # Process each tracked object
        for obj_id, obj_data in tracked_objects.items():
            # Skip if object already has a description
            if obj_id in self.object_descriptions:
                obj_data['description'] = self.object_descriptions[obj_id]
                continue

            # Skip if description request is pending
            if obj_id in self.description_pending:
                obj_data['description'] = "Processing..."
                # Check if the pending request is complete
                result = self.image_analyzer.get_pending_result(obj_id)
                if result and result["status"] == "completed":
                    self.description_pending.remove(obj_id)
                    if "error" in result:
                        description = f"Error: {result['error']}"
                    else:
                        description = result.get('description', '...')
                    self.object_descriptions[obj_id] = description
                    obj_data['description'] = description
                continue

            # Start new description request
            x1, y1, x2, y2 = map(int, obj_data['bbox'])
            obj_image = image[y1:y2, x1:x2]
            
            success, encoded_obj = cv2.imencode('.jpg', obj_image)
            if success:
                self.image_analyzer.describe_image(
                    encoded_obj.tobytes(),
                    object_id=obj_id
                )
                self.description_pending.add(obj_id)
                obj_data['description'] = "Processing..."
                logging.info(f"Started description request for object {obj_id}")
            else:
                obj_data['description'] = "Image encoding failed"

    def calculate_fps(self) -> float:
        """
        Calculate the average FPS based on frame timestamps.
        :return: Average FPS or 0 if less than 2 frames
        """
        if len(self.frames) < 2:
            return 0.0
        
        # Calculate time difference between first and last frame
        total_time = (self.frames[-1]['timestamp'] - self.frames[0]['timestamp']).total_seconds()
        if total_time == 0:
            return 0.0
        
        return (len(self.frames) - 1) / total_time


def main():
    # Create an instance of the ObjectTracker using the config file
    tracker = ObjectTracker("config.json")
    
    # Get text prompt from config, defaulting to "product."
    text_prompt = tracker.config.get("text_prompt", "product.")
    
    # Initialize webcam handler
    with WebcamHandler() as webcam:
        webcam.enable_display("Object Tracking")
        
        while True:
            # Read frame from webcam
            success, frame, fps = webcam.read_frame()
            if not success:
                logging.error("Failed to read frame from webcam")
                break
            
            # Process frame through object tracker
            tracker.process_frame(frame, text_prompt=text_prompt)
            
            # Visualize the tracked objects
            vis_frame = tracker.visualize_frame(frame, tracker.frames[-1]['objects'])
            
            # Display frame with tracking visualization
            webcam.display_frame(vis_frame, fps)
            
            # Print tracked objects and their barcodes
            for obj_id, obj_data in tracker.frames[-1]['objects'].items():
                barcode = obj_data.get('barcode', 'No barcode')
                logging.info(f"Object {obj_id}: Label = {obj_data.get('label', 'No label')}, Barcode = {barcode}")
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
