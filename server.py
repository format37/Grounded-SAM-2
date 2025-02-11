import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import logging
import time
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
# from asyncio import Semaphore
import re
import sys
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
# semaphore = Semaphore(1)  # Replace asyncio semaphore with FastAPI semaphore
# Add request counter
request_counter = 0

# Add test endpoint
@app.get("/test")
def test_endpoint():
    return {"message": "API is working!"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)  # Returns "No Content" status

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="car. tire.")
parser.add_argument("--img-path", default="notebooks/images/truck.jpg")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="outputs/test_sam2.1")
parser.add_argument("--dump-json", action="store_true", help="Enable JSON results dumping")
parser.add_argument("--force-cpu", action="store_true")

# Only parse known arguments, ignoring uvicorn-related args
args, unknown = parser.parse_known_args()

GROUNDING_MODEL = args.grounding_model
TEXT_PROMPT = args.text_prompt
IMG_PATH = args.img_path
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
DUMP_JSON_RESULTS = args.dump_json

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
logger.info("Starting to load SAM2 model...")
start_time = time.time()
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
logger.info(f"SAM2 model loaded in {time.time() - start_time:.2f} seconds")

# build grounding dino from huggingface
logger.info("Starting to load Grounding DINO model...")
start_time = time.time()
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
logger.info(f"Grounding DINO model loaded in {time.time() - start_time:.2f} seconds")

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    text_prompt: str = Form(default="car. tire.")
):
    try:
        # Update global counter
        global request_counter
        request_counter += 1
        logger.info(f">> [{request_counter}] prompt: {text_prompt}")
        
        # Read and validate the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        # logger.info(f"Image shape: {nparr.shape}")
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert OpenCV image to PIL Image
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Process text prompt
        text = text_prompt.lower()
                
        # Validate text format requirements
        if not text:
            raise HTTPException(status_code=400, detail="Text prompt cannot be empty")
        if not text.lower() == text:
            raise HTTPException(status_code=400, detail="Text prompt must be lowercase")
        if not all(q.endswith('.') for q in text.split() if q):
            raise HTTPException(status_code=400, detail="Each query must end with a dot")

        # Set image for SAM2 predictor
        sam2_predictor.set_image(np.array(image))

        # Process with Grounding DINO
        inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        # logger.info(f"Number of results: {len(results)}")
        # logger.info(f"Results: {results}")
        # Process with SAM2
        input_boxes = results[0]["boxes"].cpu().numpy()
        # logger.info(f"Input boxes: {input_boxes}")
        if len(input_boxes) > 0:
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # masks = masks.tolist()
            # Ensure masks are 2D
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            mask_rles = [single_mask_to_rle(mask) for mask in masks]
            boxes = input_boxes.tolist()
            # Convert results to desired format
            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]
        else:
            masks = []
            boxes = []
            mask_rles = []
            confidences = []
            class_names = []
        
        annotations = [
            {
                "label": class_name,
                "confidence": conf,
                "bbox": box,
                "mask": mask_rle
            }
            for class_name, conf, box, mask_rle in zip(class_names, confidences, boxes, mask_rles)
        ]
        
        return JSONResponse({"annotations": annotations})

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    process_image(IMG_PATH)

if __name__ == "__main__":
    # This block will only run when the script is run directly
    uvicorn.run(app, host="0.0.0.0", port=8765)
