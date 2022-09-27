# import all the required libraries
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import itertools
import pefile
import os, math
from queue import Queue
from threading import Thread
import argparse
from PIL import Image
from io import BytesIO
from keras.applications import imagenet_utils
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import tempfile
from tensorflow import keras
import uvicorn
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras import preprocessing
# from keras.preprocessing.image import img_to_array
from fastapi.templating import Jinja2Templates
# from tensorflow.keras.preprocessing.image import img_to_array

from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array



import sys
sys.path.append('../scripts/')
from scripts.exe_to_png import *
from scripts.preprocessPNG import *
from scripts.peDataDumpPreprocess import *
import joblib


# creating fastApi app
app_desc = """<h2> Try uploading a Portable Executable(PE) file"""
app = FastAPI(description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict")
def parse(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1]
    _, path = tempfile.mkstemp(prefix='parser_', suffix=extension)

    with open(path, 'ab') as f:
        for chunk in iter(lambda: file.file.read(10000), b''):
            f.write(chunk)

    # extract content
    content = pefile.PE(path, fast_load=True)
    img = None
    dataframe = createDataframeFromPEdump(content)
    binary_preds = getPredictions(dataframe)
    if binary_preds[1] * 100 > 60.0:
        img_path=createGreyScaleImage(path)
        img = Bayes_ResNet50_predict(img_path)
        return {'response': 'OK', 'path': path, 'predictions': binary_preds[1], 'malware': img}
    else:
        return {'Response': 'Your file is same from malware.', 'Malicious percentage': binary_preds[1] * 100,'result':'Your file is safe'}
    # remove temp file
    os.close(_)
    os.remove(path)
