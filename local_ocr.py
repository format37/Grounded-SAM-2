import cv2
import pytesseract
from PIL import Image
import os
from typing import Union, Optional

class ImageTextExtractor:
    """A class to handle OCR text extraction from images."""
    
    def __init__(self, tesseract_config: Optional[dict] = None):
        """
        Initialize the ImageTextExtractor.
        
        Args:
            tesseract_config (dict, optional): Configuration parameters for pytesseract
        """
        self.tesseract_config = tesseract_config or {}

    def preprocess_image(self, image: Union[str, bytes, cv2.Mat]) -> cv2.Mat:
        """
        Preprocess the image for better OCR results.
        
        Args:
            image: Can be a file path, bytes, or cv2 image matrix
            
        Returns:
            cv2.Mat: Preprocessed image
        """
        # Handle different input types
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image

        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised_image = cv2.fastNlMeansDenoising(gray_image)
        
        return denoised_image

    def extract_text(self, image: Union[str, bytes, cv2.Mat]) -> str:
        """
        Extract text from an image.
        
        Args:
            image: Can be a file path, bytes, or cv2 image matrix
            
        Returns:
            str: Extracted and cleaned text
        """
        processed_image = self.preprocess_image(image)
        
        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(
            processed_image, 
            **self.tesseract_config
        )
        
        # Clean up the extracted text
        cleaned_text = ' '.join(extracted_text.split())
        
        return cleaned_text

    def process_directory(self, image_dir: str, output_dir: Optional[str] = None) -> dict:
        """
        Process all images in a directory.
        
        Args:
            image_dir (str): Directory containing images
            output_dir (str, optional): Directory to save processed results
            
        Returns:
            dict: Dictionary mapping filenames to extracted text
        """
        results = {}
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        for filename in os.listdir(image_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                extracted_text = self.extract_text(image_path)
                results[filename] = extracted_text
                
        return results


def main():
    """Demonstrate the usage of ImageTextExtractor."""
    # Example usage
    image_dir = './notebooks/images/objects/'
    
    # Initialize the extractor
    extractor = ImageTextExtractor()
    
    # Process a directory of images
    results = extractor.process_directory(image_dir)
    
    # Print results
    for filename, text in results.items():
        print(f"\n=== Text from {filename} ===")
        print(text)
        print("="*50)
    
    # Example of processing a single image
    sample_image_path = os.path.join(image_dir, os.listdir(image_dir)[0])
    text = extractor.extract_text(sample_image_path)
    print(f"\nSingle image text extraction result:\n{text}")


if __name__ == "__main__":
    main()
