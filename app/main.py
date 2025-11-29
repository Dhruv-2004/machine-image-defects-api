# app/main.py
import os
import io
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
import cloudinary
import cloudinary.uploader

# ---- Config (read from environment) ----
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise RuntimeError("Set ROBOFLOW_API_KEY environment variable before starting the app.")

CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUD_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUD_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

if not (CLOUD_NAME and CLOUD_API_KEY and CLOUD_API_SECRET):
    raise RuntimeError("Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET environment variables before starting the app.")

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUD_API_KEY,
    api_secret=CLOUD_API_SECRET,
    secure=True
)

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)
MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "machine-defects/5")

app = FastAPI(title="Machine Defects Annotation + Cloudinary Upload")

# ---------- Helper: wrap text ----------
def wrap_text_to_width(text: str, font, font_scale: float, thickness: int, max_width: int) -> List[str]:
    """Greedy wrap by words so each line fits within max_width (pixels)."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        (tw, th), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if tw <= max_width:
            cur = candidate
        else:
            if cur == "":
                # single word longer than width; force-break by characters
                chunk = ""
                for ch in w:
                    cand2 = chunk + ch
                    (tw2, _), _ = cv2.getTextSize(cand2, font, font_scale, thickness)
                    if tw2 <= max_width:
                        chunk = cand2
                    else:
                        lines.append(chunk)
                        chunk = ch
                if chunk:
                    cur = chunk
                else:
                    cur = ""
            else:
                lines.append(cur)
                cur = w
    if cur:
        lines.append(cur)
    return lines

# ---------- Core annotation logic (same behaviour you approved) ----------
def annotate_with_panel(image_bgr: np.ndarray, result: dict, text_height_ratio: float = 0.04) -> (np.ndarray, List[str]):
    """
    Annotate bounding boxes on image and add a centered white panel below with wrapped, centered labels.
    Returns (final_image, labels_list)
    """
    preds = result.get("predictions", [])
    if len(preds) == 0:
        annotated = image_bgr.copy()
        labels = ["No detections"]
    else:
        xyxy, conf_list, class_ids, labels = [], [], [], []
        for p in preds:
            cx, cy = p["x"], p["y"]
            w, h = p["width"], p["height"]

            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2

            xyxy.append([x1, y1, x2, y2])
            conf_list.append(float(p["confidence"]))
            class_ids.append(int(p.get("class_id", 0)))
            labels.append(f"{p['class']} ({p['confidence']:.2f})")  # Text below image

        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(conf_list),
            class_id=np.array(class_ids)
        )

        # Draw bounding boxes (no labels on image)
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    # Prepare full text from labels
    if isinstance(labels, (list, tuple)):
        full_text = ", ".join(labels)
    else:
        full_text = str(labels)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = annotated.shape[:2]

    # Scale font relative to image height so text appears consistent across different image sizes
    desired_text_h = max(12, int(round(img_h * text_height_ratio)))
    (base_w, base_h), base_baseline = cv2.getTextSize("Ay", font, 1.0, 1)
    font_scale = desired_text_h / float(base_h)
    thickness = max(1, int(round(font_scale)))

    # Wrap text to multiple lines to fit the image width
    h_margin = 20
    max_text_width = max(100, img_w - 2 * h_margin)
    lines = wrap_text_to_width(full_text, font, font_scale, thickness, max_text_width)

    # Recompute line heights and panel height
    (text_w_sample, text_h), baseline = cv2.getTextSize("Ay", font, font_scale, thickness)
    line_spacing = max(4, int(0.2 * text_h))
    panel_padding = 12
    panel_height = panel_padding * 2 + len(lines) * text_h + (len(lines) - 1) * line_spacing
    panel_height = max(panel_height, 40)

    # Create white panel and draw centered lines
    panel = np.ones((panel_height, img_w, 3), dtype=np.uint8) * 255
    y = panel_padding + text_h
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (img_w - tw) // 2
        cv2.putText(panel, line, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += text_h + line_spacing

    final_image = np.vstack([annotated, panel])
    return final_image, labels

# ---------- Endpoint: upload-image (calls Roboflow, annotates, uploads to Cloudinary) ----------
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts image file, runs Roboflow inference, annotates image (boxes + centered panel below),
    saves to PNG, uploads to Cloudinary, returns JSON with labels and uploaded URL.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Save upload to a temp file (Roboflow inference client expects a path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name

    try:
        # Call Roboflow inference (path-based)
        result = CLIENT.infer(tmp_path, model_id=MODEL_ID)

        # Convert bytes to cv2 image
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # Annotate image using the same logic you approved
        final_img, labels = annotate_with_panel(img, result, text_height_ratio=0.04)

        # Save final image to a temporary PNG for Cloudinary upload
        out_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        out_path = out_tmp.name
        out_tmp.close()  # will write using cv2.imwrite

        # cv2.imwrite expects BGR image; save with high quality
        cv2.imwrite(out_path, final_img)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(out_path, resource_type="image")
        uploaded_url = upload_result.get("secure_url") or upload_result.get("url")

        # Return JSON with labels and URL
        return JSONResponse({
            "status": "success",
            "labels": labels,
            "url": uploaded_url,
            "raw_inference": result  # optional: remove if you don't want to return full raw response
        })

    except Exception as e:
        # bubble up some useful debug info but avoid exposing secrets
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
    finally:
        # cleanup temp files if they exist
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            if 'out_path' in locals() and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass

# health endpoint
@app.get("/")
def root():
    return {"status": "ok", "message": "Send POST /upload-image/ with form field 'file'."}

# If run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
