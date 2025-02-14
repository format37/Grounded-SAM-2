#!/usr/bin/env python3
"""
Barcode reader module for processing and decoding barcodes from images.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from pyzbar.pyzbar import decode, Decoded
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BarcodeResult:
    """Data class to store barcode detection results."""
    data: str
    type: str
    location: tuple
    confidence: float = 0.0

class BarcodeReader:
    """Class for handling barcode reading operations."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the BarcodeReader.
        
        Args:
            debug (bool): Enable debug mode for additional visualization
        """
        self.debug = debug
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better barcode detection.
        Converts to grayscale and normalizes brightness/contrast.
        """
        try:
            # Convert to grayscale if necessary
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Normalize brightness and contrast
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            normalized = clahe.apply(gray)

            # Save the processed image if in debug mode
            if self.debug:
                # Create directory if it doesn't exist
                save_dir = Path("./notebooks/images/processed_objects")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save both grayscale and normalized images
                timestamp = cv2.getTickCount()
                gray_path = save_dir / f"grayscale_{timestamp}.jpg"
                norm_path = save_dir / f"normalized_{timestamp}.jpg"
                cv2.imwrite(str(gray_path), gray)
                cv2.imwrite(str(norm_path), normalized)
                logger.info(f"Saved processed images to: {save_dir}")

            return normalized

        except Exception as e:
            logger.error(f"Error during image preprocessing: {str(e)}")
            raise
    
    def read_barcode(self, image: Union[str, Path, np.ndarray]) -> List[BarcodeResult]:
        """
        Read barcodes from an image.
        
        Args:
            image: Can be a file path (str or Path) or numpy array
            
        Returns:
            List[BarcodeResult]: List of detected barcodes with their information
        """
        try:
            # Handle different input types
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Detect and decode barcodes
            barcodes = decode(processed_image)
            
            results = []
            for barcode in barcodes:
                result = BarcodeResult(
                    data=barcode.data.decode('utf-8'),
                    type=barcode.type,
                    location=barcode.rect
                )
                results.append(result)
                logger.info(f"Detected barcode: {result}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error reading barcode: {str(e)}")
            raise

def main():
    """Example usage of the BarcodeReader class."""
    # Example image path
    image_folder = "./notebooks/images/objects"
    
    try:
        # Get the latest barcode image from the folder
        barcode_files = [f for f in os.listdir(image_folder) if 'barcode' in f.lower()]
        if not barcode_files:
            raise FileNotFoundError("No barcode images found in the specified folder")
        
        latest_file = max(barcode_files, key=lambda x: os.path.getctime(os.path.join(image_folder, x)))
        image_path = os.path.join(image_folder, latest_file)

        logger.info(f"Using image: {image_path}")
        
        # Initialize reader
        reader = BarcodeReader(debug=True)
        
        # Read barcodes
        results = reader.read_barcode(image_path)
        
        # Print results
        if results:
            print("\nDetected Barcodes:")
            for idx, result in enumerate(results, 1):
                print(f"\nBarcode {idx}:")
                print(f"Data: {result.data}")
                print(f"Type: {result.type}")
                print(f"Location: {result.location}")
        else:
            print("No barcodes detected in the image.")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 