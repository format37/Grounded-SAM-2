import cv2
import logging
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import time
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebcamHandler:
    """
    A class to handle webcam operations with configurable settings.
    """
    
    SUPPORTED_RESOLUTIONS = {
        "3MP": (2048, 1536),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "VGA": (640, 480),
        "QVGA": (320, 240)
    }

    def __init__(self, config_path: str = "config.json", camera_id: int = 0):
        """
        Initialize the WebcamHandler.

        Args:
            config_path (str): Path to the configuration file
            camera_id (int): ID of the camera to use (default is 0 for primary webcam)
        """
        self.config = self._load_config(config_path)
        self.camera_id = camera_id
        self.cap = None
        self.frame_counter = 0
        self.prev_frame_time = 0
        self.current_resolution = None
        self.window_name = 'Webcam Feed'
        self.display_enabled = False

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path (str): Path to the configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config file: {e}")
            return {
                "camera": {
                    "default_resolution": "720p",
                    "resolutions": WebcamHandler.SUPPORTED_RESOLUTIONS
                }
            }

    def start(self) -> bool:
        """
        Start the webcam capture.

        Returns:
            bool: True if webcam started successfully, False otherwise
        """
        # Check if the camera is already started
        if self.cap is not None and self.cap.isOpened():
            logger.info("Webcam already started.")
            return True

        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Could not open webcam {self.camera_id}")
                return False

            # Set resolution from config
            self._set_resolution()
            
            # Verify camera is working by reading a test frame
            ret, _ = self.cap.read()
            if not ret:
                logger.error("Could not read frame from camera")
                self.cap.release()
                return False
            
            logger.info(f"Webcam started successfully with resolution: {self.current_resolution}")
            return True
        except Exception as e:
            logger.error(f"Error starting webcam: {e}")
            return False

    def _set_resolution(self) -> None:
        """Set the webcam resolution based on configuration."""
        camera_config = self.config.get('camera', {})
        default_resolution = camera_config.get('default_resolution', '720p')
        
        if default_resolution not in self.SUPPORTED_RESOLUTIONS:
            logger.warning(f"Invalid resolution {default_resolution}, defaulting to 720p")
            width, height = self.SUPPORTED_RESOLUTIONS["720p"]
        else:
            width, height = self.SUPPORTED_RESOLUTIONS[default_resolution]

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_resolution = (actual_width, actual_height)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read a frame from the webcam with FPS calculation.

        Returns:
            Tuple[bool, Optional[np.ndarray], float]: Success flag, frame data, and current FPS
        """
        if not self.cap or not self.cap.isOpened():
            return False, None, 0.0

        ret, frame = self.cap.read()
        if not ret:
            return False, None, 0.0

        self.frame_counter += 1
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time

        return True, frame, fps

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current resolution of the webcam.

        Returns:
            Tuple[int, int]: Current width and height of the webcam feed
        """
        return self.current_resolution

    def enable_display(self, window_name: str = 'Webcam Feed') -> None:
        """
        Enable display window for the video feed.

        Args:
            window_name (str): Name of the display window
        """
        self.window_name = window_name
        self.display_enabled = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def disable_display(self) -> None:
        """Disable display window."""
        if self.display_enabled:
            cv2.destroyWindow(self.window_name)
            self.display_enabled = False

    def display_frame(self, frame: np.ndarray, fps: float = None) -> None:
        """
        Display a frame in the window.

        Args:
            frame (np.ndarray): Frame to display
            fps (float, optional): FPS to display on frame
        """
        if not self.display_enabled:
            return

        display_frame = frame.copy()
        
        # Add FPS if provided
        if fps is not None:
            cv2.putText(
                display_frame,
                f'FPS: {fps:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(self.window_name, display_frame)

    def release(self) -> None:
        """Release the webcam resources."""
        self.disable_display()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            logger.info("Webcam released")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

def main():
    """Example usage of the WebcamHandler class."""
    with WebcamHandler() as webcam:
        # The webcam is already started in the context manager (__enter__)
        webcam.enable_display("Live Feed")

        while True:
            success, frame, fps = webcam.read_frame()
            if not success:
                break

            # Process/modify the frame if desired (here we simply copy it)
            modified_frame = frame.copy()
            
            # Display the frame in the enabled window
            webcam.display_frame(modified_frame, fps)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main() 