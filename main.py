from model import *
from camera_api import *
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
cam = Cameras()
model = Model()


@app.get("/")
def root():
    return {"Hello": "World"}


@app.get("/get_value")
def get_value():
    img1, img2 = cam.get_screenshots()
    value, log = model.calculate_people(img1, img2)
    return JSONResponse({"value": value, "log": log})
