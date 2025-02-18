#!/usr/bin/env python3
import logging
import cv2
from object_tracker import ObjectTracker
from webcam_handler import WebcamHandler
from pydantic import BaseModel, Field
from typing import Literal, List, Union, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

class ObjectDescription(BaseModel):
    description: str
    description_confidence_01: float
    department: Literal["tea", "milk", "battery", "tools", "other"]
    form: Literal["parallelepiped", "cylinder", "sphere", "other"]
    filling: Literal["liquid", "solid", "empty", "other"]
    weight_grams: int

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
