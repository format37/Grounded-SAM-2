# Grounded SAM 2: Open-WorldObject Detection & Mask Generation

This is a Docker server, that can accept images and text prompts, and return the object detection and mask generation results.

## Installation
```
git clone https://github.com/format37/Grounded-SAM-2
cd Grounded-SAM-2
```
## Docker:
Ensure that you have installed:
* docker  
* nvidia-container-toolkit  
* nvcc  
```
docker compose build
sh compose.sh
```

## Client example
```
python client.py
```

# Based on
[IDEA-Research Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)