import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Function for Non-Deep Learning Models
def prepare_ml_data(data_path, target_size=(64, 64)):
    filenames = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
    X, y = [], []
    for filename in filenames:
        img = Image.open(os.path.join(data_path, filename)).convert('RGB').resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        X.append(img_array.flatten())
        y.append(0 if filename.split('.')[0] == 'cat' else 1)
    return np.array(X), np.array(y)


#Function for Deep Learning Models
def prepare_dl_data(data_path, target_size=(224, 224)):
    filenames = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
    X, y = [], []
    for filename in filenames:
        img = Image.open(os.path.join(data_path, filename)).convert('RGB').resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        X.append(img_array)
        y.append(0 if filename.split('.')[0] == 'cat' else 1)
    return np.array(X), np.array(y)
