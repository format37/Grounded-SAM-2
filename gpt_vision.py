import base64
import requests
import json
import time
from PIL import Image
import io
from typing import Dict, List, Optional, Set
import logging
import asyncio
import concurrent.futures
import threading

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

        # Add tracking for pending requests
        self.pending_requests = {}
        self.pending_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=30)

        # Add set to track current objects
        self.current_objects = set()

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

    def _describe_image_sync(self, image_data: bytes, prompt: str) -> Dict:
        """Synchronous version of describe_image"""
        try:
            # Convert binary data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            image_url = self._process_image(image)
            
            messages = []
            self._append_message(messages, "user", prompt, image_url)

            start_time = time.time()
            logger.info("sending request to openai")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 500
                }
            )
            if not response.ok:
                logger.error(f"OpenAI API request failed with status {response.status_code}. Response: {response.text}")
                response.raise_for_status()
            
            elapsed_time = time.time() - start_time
            
            response_json = response.json()
            description = response_json['choices'][0]['message']['content']
            logger.info(f"received response from openai: {description}")

            return {
                "description": description,
                "elapsed_time": elapsed_time,
                "cached": False
            }
            
        except Exception as e:
            return {"error": str(e)}

    def describe_image(self, image_data: bytes, object_id: int = None, 
                      prompt: str = "Please classify in one short sentence what you see in this image?") -> Dict:
        """Asynchronous image description - returns immediately with status"""
        logger.debug(f"Processing image for object {object_id}")
        
        # Check cache first
        if object_id is not None and object_id in self.object_descriptions:
            logger.debug(f"Using cached description for object {object_id}")
            return {
                "description": self.object_descriptions[object_id],
                "elapsed_time": 0,
                "cached": True,
                "status": "completed"
            }

        # Check if request is already pending
        with self.pending_lock:
            if object_id in self.pending_requests:
                logger.debug(f"Request pending for object {object_id}")
                return {"status": "pending"}

            # Submit new request
            logger.info(f"Starting new request for object {object_id}")
            future = self.executor.submit(self._describe_image_sync, image_data, prompt)
            self.pending_requests[object_id] = future
            return {"status": "pending"}

    def get_pending_result(self, object_id: int) -> Optional[Dict]:
        """Check if a pending request is complete"""
        with self.pending_lock:
            if object_id not in self.pending_requests:
                return None
                
            future = self.pending_requests[object_id]
            if not future.done():
                return {"status": "pending"}
                
            try:
                # Get result and cleanup
                result = future.result()
                del self.pending_requests[object_id]
                
                # Cache successful result
                if "error" not in result:
                    self.object_descriptions[object_id] = result["description"]
                    logger.info(f"Cached new description for object {object_id}: {result['description']}")
                
                return {**result, "status": "completed"}
            except Exception as e:
                logger.error(f"Error getting result for object {object_id}: {e}")
                del self.pending_requests[object_id]
                return {"error": str(e), "status": "completed"}

    def cleanup_old_objects(self, current_object_ids: Set[int]) -> None:
        """Remove cached descriptions for objects that no longer exist"""
        # Only clean up if the set of objects has changed
        if current_object_ids != self.current_objects:
            # Update current objects set
            self.current_objects = set(current_object_ids)
            
            # Clean up cached descriptions
            to_remove = [obj_id for obj_id in self.object_descriptions 
                        if obj_id not in self.current_objects]
            for obj_id in to_remove:
                del self.object_descriptions[obj_id]
            
            # Clean up pending requests
            with self.pending_lock:
                pending_to_remove = [obj_id for obj_id in self.pending_requests 
                                   if obj_id not in self.current_objects]
                for obj_id in pending_to_remove:
                    # Cancel future if possible
                    future = self.pending_requests[obj_id]
                    future.cancel()
                    del self.pending_requests[obj_id]
            
            if to_remove or pending_to_remove:
                logger.info(f"Cleaned up {len(to_remove)} cached and {len(pending_to_remove)} pending descriptions")

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