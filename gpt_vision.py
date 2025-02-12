import base64
import requests
import json
import time
from PIL import Image
import io
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    # Maximum dimensions for resizing images
    MAX_WIDTH = 320
    MAX_HEIGHT = 320

    def __init__(self, api_key: str = None, config_path: str = "config.json"):
        """
        Initialize the ImageAnalyzer with API configuration.
        
        Args:
            api_key: Optional API key to use. If not provided, will read from config file
            config_path: Path to config file (used if api_key not provided)
        """
        with open(config_path) as f:
            config = json.load(f)
        if api_key:
            self.api_key = api_key.strip()
            
        else:
            self.api_key = config["api_key"].strip()
        self.model = config["model"]

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Add cache for object descriptions
        self.object_descriptions = {}

    def _append_message(self, messages: List[Dict], role: str, text: str, 
                       image_url: str, detail: str = "low") -> None:
        """Append a message to the conversation history."""
        messages.append({
            "role": role,
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": detail
                    },
                },
            ],
        })

    def _process_image(self, image: Image.Image) -> str:
        """Process and encode image maintaining aspect ratio."""
        ratio = min(self.MAX_WIDTH/image.width, self.MAX_HEIGHT/image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_img = image.resize(new_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        resized_img.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"

    def describe_image(self, image_data: bytes, object_id: int = None, prompt: str = "Please classify in one short sentence what you see in this image?") -> Dict:
        logger.info(f"describing image: {object_id}")
        """Analyze image from binary data and return description."""
        # Check cache if object_id is provided
        if object_id is not None and object_id in self.object_descriptions:
            logger.info(f"Using cached description for object {object_id}")
            return {
                "description": self.object_descriptions[object_id],
                "elapsed_time": 0,
                "cached": True
            }

        try:
            # Convert binary data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            image_url = self._process_image(image)
            
            messages = []
            self._append_message(messages, "user", prompt, image_url)

            start_time = time.time()
            logger.info(f"sending request to openai")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 2000
                }
            )
            if not response.ok:
                logger.error(f"OpenAI API request failed with status {response.status_code}. Response: {response.text}")
                # Optionally raise an exception or handle the error case
                response.raise_for_status()
            
            elapsed_time = time.time() - start_time
            
            response_json = response.json()
            description = response_json['choices'][0]['message']['content']
            logger.info(f"received response from openai: {description}")
            # Cache the result if object_id is provided
            if object_id is not None:
                self.object_descriptions[object_id] = description

            logger.info(f"New GPT Vision analysis for object {object_id}: {description}")

            return {
                "description": description,
                "elapsed_time": elapsed_time,
                "cached": False
            }
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Example usage of the ImageAnalyzer class."""
    # Initialize analyzer
    analyzer = ImageAnalyzer()
    
    # Read image file as binary
    image_path = "./notebooks/images/objects/image_1_product_0.70.png"
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Get description
    result = analyzer.describe_image(image_data)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Description: {result['description']}")
        print(f"Processing time: {result['elapsed_time']:.2f} seconds")

if __name__ == "__main__":
    main()