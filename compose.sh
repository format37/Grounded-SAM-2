# docker compose build
docker compose down
# docker compose run --rm --name sam grounded_sam bash
# -v "$(pwd)/grounded_sam2_hf_loop.py:/home/appuser/Grounded-SAM-2/grounded_sam2_hf_loop.py" \
docker compose run --rm --name sam \
    -v "$(pwd)/client.py:/home/appuser/Grounded-SAM-2/client.py" \
    -v "$(pwd)/server.py:/home/appuser/Grounded-SAM-2/server.py" \
    -v "$(pwd)/notebooks/images:/home/appuser/Grounded-SAM-2/notebooks/images" \
    grounded_sam uvicorn server:app --host 0.0.0.0 --port 8765
# docker compose run --rm --name sam grounded_sam