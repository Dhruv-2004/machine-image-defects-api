# Machine Defects Annotation API

FastAPI service that accepts an uploaded image, calls a Roboflow Inference model, draws bounding boxes, and returns an annotated PNG image with a centered label panel below the image.

## Local testing

1. Set env var:
```bash
export ROBOFLOW_API_KEY="your_key_here"
