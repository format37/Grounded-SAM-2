import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    """Class to store tracked object data"""
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    frame_idx: int
    label: str
    confidence: float

class ObjectTracker:
    def __init__(self, max_frames: int = 5, distance_threshold: float = 50.0):
        """
        Initialize the object tracker
        
        Args:
            max_frames: Number of previous frames to consider
            distance_threshold: Maximum distance between boxes to be considered same object
        """
        self.max_frames = max_frames
        self.distance_threshold = distance_threshold
        self.tracked_objects: List[TrackedObject] = []
        self.next_id = 1
        self.current_frame = 0
    
    def calculate_box_distance(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate distance between two bounding boxes using corner points
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: Average distance between corners
        """
        # Calculate distances between corresponding corners
        corners1 = [(box1[0], box1[1]), (box1[2], box1[3])]  # Top-left, bottom-right
        corners2 = [(box2[0], box2[1]), (box2[2], box2[3])]
        
        distances = []
        for c1, c2 in zip(corners1, corners2):
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            distances.append(dist)
        
        return np.mean(distances)
    
    def find_matching_object(self, bbox: List[float], label: str) -> Optional[int]:
        """
        Find matching object from previous frames
        
        Args:
            bbox: Current bounding box
            label: Object label
            
        Returns:
            Optional[int]: ID of matching object if found, None otherwise
        """
        # Look through recent frames
        min_frame = max(0, self.current_frame - self.max_frames)
        recent_objects = [
            obj for obj in self.tracked_objects 
            if obj.frame_idx >= min_frame and obj.label == label
        ]
        
        if not recent_objects:
            return None
        
        # Calculate distances to all recent objects
        distances = [
            (obj.id, self.calculate_box_distance(bbox, obj.bbox))
            for obj in recent_objects
        ]
        
        # Find closest object within threshold
        closest_id = None
        min_distance = float('inf')
        
        for obj_id, distance in distances:
            if distance < self.distance_threshold and distance < min_distance:
                min_distance = distance
                closest_id = obj_id
        
        return closest_id
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracking for new frame
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'label', 'confidence'
            
        Returns:
            List[Dict]: Updated detections with tracking IDs
        """
        self.current_frame += 1
        updated_detections = []
        
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            confidence = det['confidence']
            
            # Try to find matching object
            matching_id = self.find_matching_object(bbox, label)
            
            if matching_id is None:
                # New object
                matching_id = self.next_id
                self.next_id += 1
            
            # Store tracked object
            self.tracked_objects.append(TrackedObject(
                id=matching_id,
                bbox=bbox,
                frame_idx=self.current_frame,
                label=label,
                confidence=confidence
            ))
            
            # Update detection with ID
            det_with_id = det.copy()
            det_with_id['label'] = f"[{matching_id}] {label}"
            updated_detections.append(det_with_id)
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        return updated_detections
    
    def _cleanup_old_tracks(self):
        """Remove tracks from frames older than max_frames"""
        min_frame = self.current_frame - self.max_frames
        self.tracked_objects = [
            obj for obj in self.tracked_objects 
            if obj.frame_idx >= min_frame
        ]

def main():
    """Demonstrate the usage of ObjectTracker"""
    # Create sample detections for multiple frames
    frame1_detections = [
        {'bbox': [100, 100, 200, 200], 'label': 'car', 'confidence': 0.9},
        {'bbox': [300, 300, 400, 400], 'label': 'person', 'confidence': 0.8}
    ]
    
    frame2_detections = [
        {'bbox': [110, 110, 210, 210], 'label': 'car', 'confidence': 0.85},  # Slightly moved
        {'bbox': [500, 500, 600, 600], 'label': 'person', 'confidence': 0.75}  # New position
    ]
    
    # Initialize tracker
    tracker = ObjectTracker(max_frames=5, distance_threshold=50.0)
    
    # Process frames
    print("Frame 1 results:")
    results1 = tracker.update(frame1_detections)
    for det in results1:
        print(f"Detection: {det}")
    
    print("\nFrame 2 results:")
    results2 = tracker.update(frame2_detections)
    for det in results2:
        print(f"Detection: {det}")

if __name__ == "__main__":
    main() 