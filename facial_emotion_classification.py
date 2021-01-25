import uvicorn
from fastapi import FastAPI,File,UploadFile
from pydantic import BaseModel
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np
from fer import FER
# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
detector = FER(mtcnn=True)
class Symptom(BaseModel):
    fever: bool=False
    dry_cough: bool=False
    tiredness: bool=False
    breathing_problem: bool=False

def get_risk_level(symptom: Symptom):
    if not (symptom.fever or symptom.dry_cough or symptom.tiredness or symptom.breathing_problem):
        return 'Low risk level. THIS IS A DEMO APP'

    if not (symptom.breathing_problem or symptom.dry_cough):
        if symptom.fever:
            return 'moderate risk level. THIS IS A DEMO APP'

    if symptom.breathing_problem:
        return 'High-risk level. THIS IS A DEMO APP'

    return 'THIS IS A DEMO APP'

def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("model loaded")
    return model

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def predict(image: Image.Image):

    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0

    result = decode_predictions(model.predict(image), 2)[0]

    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"

        response.append(resp)

    return response

def predict_emotion(image: Image.Image):
    image = np.asarray(image)[..., :3]
    result = detector.detect_emotions(image)
    if len(result) >0:
        emotions = result[0]["emotions"]
        print(emotions)
        return emotions
    else:
        return "Emotion not detected"

app = FastAPI()
model = load_model()
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction

@app.post("/api/covid-symptom-check")
def check_risk(symptom: Symptom):
    return get_risk_level(symptom)

@app.post("/api/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/api/emotion_detection/")
async def check_emotion(file: UploadFile = File(...)):
    print("hello")
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    return predict_emotion(image)

if __name__ == "__main__":
    try:
        uvicorn.run(app, debug=True,port=8080)
    except Exception as err:
        print("OS error: {0}".format(err))