from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from io import BytesIO
from typing import Optional
from keras._tf_keras.keras.models import load_model
import google.generativeai as genai
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# @app.get("/", response_class=HTMLResponse)
# async def frontend(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join("templates", "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)



# Load model
model = load_model('enhanced_model.keras')

# Input shapes
XRAY_SHAPE = (224, 224, 1)
DEXA_SHAPE = (224, 224, 1)

# Configure Gemini
genai.configure(api_key="AIzaSyCb_9hU4RC8AoiI0UHP1B0KlwzaJ2I76jk")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def image_to_npy(upload_file: UploadFile, target_shape):
    img_bytes = np.frombuffer(upload_file.file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_shape[:2])
    img = img.astype(np.float16) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def override_prediction(class_probs: list[float]) -> int:
    for i in range(2):
        base = "{:.2e}".format(class_probs[i])
        base_value = float(base.split('e')[0])
        if base_value > 3.0:
            return i
    return 2

def generate_medical_report(predicted_class: int, class_probs: list[float]) -> str:
    prompt = f"""
    You are a clinical expert. Based on this AI model prediction:

    Predicted class: {predicted_class}
    Class probabilities: {class_probs}

    Class 0: Normal bone density
    Class 1: Osteopenia
    Class 2: Osteoporosis

    Generate a detailed medical report with interpretation and recommendations.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Failed to generate report: {str(e)}"

def analyze_bmd_pdf(upload_file: UploadFile) -> dict:
    try:
        # Read PDF into bytes
        pdf_bytes = upload_file.file.read()

        # Upload PDF to Gemini
        file = genai.upload_file(
            path=BytesIO(pdf_bytes),
            display_name=upload_file.filename,
            mime_type="application/pdf"
        )

        print(f"Uploaded file to Gemini: {file.uri}")

        # Compose prompt
        prompt = """
        You are a medical expert. Carefully review this BMD (Bone Mineral Density) report in PDF format.
        Based on the scan report, determine the condition using these classes:
        - Class 0: Normal
        - Class 1: Osteopenia
        - Class 2: Osteoporosis

        Then explain the reasoning behind the diagnosis using appropriate medical terms and metrics like T-scores or Z-scores.
        Respond clearly with:
        - Predicted class (as number)
        - Medical summary
        """

        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content([prompt, file])

        return {
            "bmd_prediction_raw_text": response.text.strip()
        }

    except Exception as e:
        return {"bmd_analysis_error": str(e)}


@app.post("/predict")
async def predict(
    xray: Optional[UploadFile] = File(None),
    dexa: Optional[UploadFile] = File(None),
    bmd_report: Optional[UploadFile] = File(None)
):
    try:
        results = {}

        # If BMD report is provided, analyze using Gemini
        if bmd_report:
            results.update(analyze_bmd_pdf(bmd_report))

        # If X-ray or DEXA is provided, proceed with model-based prediction
        if xray or dexa:
            xray_input = image_to_npy(xray, XRAY_SHAPE) if xray else np.zeros((1, *XRAY_SHAPE), dtype='float32')
            dexa_input = image_to_npy(dexa, DEXA_SHAPE) if dexa else np.zeros((1, *DEXA_SHAPE), dtype='float32')

            prediction = model.predict([xray_input, dexa_input])
            class_probs = prediction[0].tolist()
            predicted_class = override_prediction(class_probs)
            confidence = float(np.max(prediction))
            medical_report = generate_medical_report(predicted_class, class_probs)

            results.update({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probs": class_probs,
                "medical_report": medical_report
            })

        if not results:
            return JSONResponse({"error": "At least one input (X-ray, DEXA, or BMD PDF) must be provided."}, status_code=400)

        return results

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
