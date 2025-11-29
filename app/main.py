# app/main.py
import os
import io
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cloudinary
import cloudinary.uploader

# ---------------------------
# Configuration (use env vars)
# ---------------------------
# It's strongly recommended to set these in your deployment environment (Render / Docker)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

# If you previously hard-coded credentials and want to keep that for local quick testing,
# you can temporarily uncomment the following lines and fill them in (NOT recommended for repos):
# ROBOFLOW_API_KEY = ROBOFLOW_API_KEY or "ByAqdBr5LXCiSreNcpSV"
# CLOUDINARY_CLOUD_NAME = CLOUDINARY_CLOUD_NAME or "dk7te6qfq"
# CLOUDINARY_API_KEY = CLOUDINARY_API_KEY or "544246722266915"
# CLOUDINARY_API_SECRET = CLOUDINARY_API_SECRET or "IE_ePyypf-5qULT_IwL69o3wEh8"

if not ROBOFLOW_API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY environment variable is required.")

if not (CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET):
    # We allow missing Cloudinary creds to keep local dev ease, but warn. Upload will fail without them.
    print("Warning: Cloudinary credentials not fully set. Upload to cloudinary will fail unless set.")

# Configure Cloudinary (safe even if env vars are present)
cloudinary.config(
  cloud_name=CLOUDINARY_CLOUD_NAME,
  api_key=CLOUDINARY_API_KEY,
  api_secret=CLOUDINARY_API_SECRET,
  secure=True
)

# Roboflow inference client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

MODEL_ID = os.environ.get("MODEL_ID", "machine-defects/5")

app = FastAPI(title="Machine Defects Annotation API (Roboflow + Cloudinary)")

# ---------------------------
# Your original annotation logic (kept same semantics)
# ---------------------------
def annotate_roboflow_result(image_path):
    """
    This preserves your original logic:
    - Calls CLIENT.infer on the given image_path
    - Converts Roboflow center-x,y,width,height to xyxy and builds supervision.Detections
    - Annotates bounding boxes and returns annotated image + labels list
    """
    result = CLIENT.infer(image_path, model_id=MODEL_ID)

    # Load image
    image = cv2.imread(image_path)

    if not result.get('predictions'):
        print(f"No detections found in: {image_path}")
        return image, []  # Return original image and empty labels

    xyxy = []
    confidences = []
    class_ids = []
    labels = []

    for prediction in result['predictions']:
        x_center, y_center = prediction['x'], prediction['y']
        width, height = prediction['width'], prediction['height']

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        xyxy.append([x_min, y_min, x_max, y_max])
        confidences.append(prediction['confidence'])
        class_ids.append(prediction.get('class_id', 0))

        label = f"{prediction['class']} {prediction['confidence']:.2f}"
        labels.append(label)

    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidences),
        class_id=np.array(class_ids)
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Draw boxes and (optionally) inline labels (kept because it was in your original code).
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated, labels

# ---------------------------
# Helper: wrap & draw centered panel below image
# (We use the improved panel approach but keep your annotate_roboflow_result logic intact)
# ---------------------------
def wrap_text_to_width(text, font, font_scale, thickness, max_width):
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
                # single long word: break into character chunks
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

def add_centered_panel_below(annotated_bgr: np.ndarray, labels):
    # Combine labels into a single string
    if isinstance(labels, (list, tuple)):
        full_text = ", ".join(labels) if labels else "No detections"
    else:
        full_text = str(labels or "No detections")

    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = annotated_bgr.shape[:2]

    # scale text relative to image height for consistent appearance
    text_height_ratio = 0.04  # 4% of image height (tweak if needed)
    desired_text_h = max(12, int(round(img_h * text_height_ratio)))
    (base_w, base_h), base_baseline = cv2.getTextSize("Ay", font, 1.0, 1)
    font_scale = desired_text_h / float(base_h)
    thickness = max(1, int(round(font_scale)))

    h_margin = 20
    max_text_width = img_w - 2 * h_margin
    lines = wrap_text_to_width(full_text, font, font_scale, thickness, max_text_width)

    (text_w_sample, text_h), baseline = cv2.getTextSize("Ay", font, font_scale, thickness)
    line_spacing = max(4, int(0.2 * text_h))
    panel_padding = 12
    panel_height = panel_padding * 2 + len(lines) * text_h + (len(lines) - 1) * line_spacing
    panel_height = max(panel_height, 40)

    panel = np.ones((panel_height, img_w, 3), dtype=np.uint8) * 255  # white panel

    y = panel_padding + text_h
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (img_w - tw) // 2
        cv2.putText(panel, line, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += text_h + line_spacing

    final = np.vstack([annotated_bgr, panel])
    return final

# ---------------------------
# API endpoint (keeps your upload-image route behavior)
# ---------------------------
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # write incoming file to a temporary path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name

        # Use your original function to annotate and get labels
        annotated_img, labels = annotate_roboflow_result(tmp_path)
        print("Labels:", labels)

        # Instead of matplotlib overlay, create a combined image with a centered panel below
        final_img = add_centered_panel_below(annotated_img, labels)

        # Save to temp PNG for upload
        out_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        out_tmp_path = out_tmp.name
        out_tmp.close()
        # write PNG (high quality)
        cv2.imwrite(out_tmp_path, final_img)

        # Upload to Cloudinary IF credentials are available
        upload_url = None
        upload_result = None
        if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
            try:
                upload_result = cloudinary.uploader.upload(out_tmp_path)
                upload_url = upload_result.get("url")
                print("Uploaded to Cloudinary:", upload_url)
            except Exception as e:
                print("Cloudinary upload failed:", str(e))
        else:
            print("Cloudinary credentials missing: skipping upload.")

        # Clean up temporary files (best-effort)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        try:
            os.remove(out_tmp_path)
        except Exception:
            pass

        return JSONResponse({
            "status": "success",
            "labels": labels,
            "url": upload_url,
            "cloudinary_response": upload_result
        })

    except Exception as e:
        # preserve your original behavior of printing error
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Run for local debugging
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
