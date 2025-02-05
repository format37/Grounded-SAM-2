from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import torch
import logging
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pycocotools.mask as mask_util
import io
import uvicorn
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Initialize models globally
logger.info("Loading models...")
# Initialize SAM2
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Initialize Grounding DINO
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

app = FastAPI()

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

async def process_image(image_bytes: bytes, text_prompt: str) -> dict:
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    image_rgb = image.convert("RGB")
    
    # Set image for SAM2
    sam2_predictor.set_image(np.array(image_rgb))
    
    # Process with Grounding DINO
    inputs = processor(images=image_rgb, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    # Get boxes for SAM2
    input_boxes = results[0]["boxes"].cpu().numpy()
    
    # Get masks from SAM2
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    # Ensure masks are 2D
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # Convert results to desired format
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    boxes = input_boxes.tolist()
    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    
    annotations = [
        {
            "label": class_name,
            "confidence": conf,
            "bbox": box,
            "mask": mask_rle
        }
        for class_name, conf, box, mask_rle in zip(class_names, confidences, boxes, mask_rles)
    ]
    
    return {"annotations": annotations}

@app.post("/process-image/")
async def process_image_endpoint(
    file: UploadFile = File(...),
    text_prompt: str = "car. tire."  # Default prompt
):
    contents = await file.read()
    results = await process_image(contents, text_prompt)
    return JSONResponse(content=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
