# Preprocessing script for image classification vignette
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

# XG Boost

import sys

# Looks in the directory above for the src folder
sys.path.append("../src")  

# Calling the preprocessing function
from scripts.preprocessing import prepare_ml_data

# Import necessary libraries for modeling
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

# Prepare data for classical ML models (XGBoost, SVM)
X, y = prepare_ml_data("data_sample", target_size=(64, 64))

# Train-test split
import os
import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y, # ensures same proportion of cats and dogs in train and test
    random_state=42
)

# Optional: Save processed arrays for reproducibility

# Define a directory where processed data will be stored
#save_dir = "../data/processed"
#os.makedirs(save_dir, exist_ok=True)   

# Save the train/test splits as .npy files
#np.save(os.path.join(save_dir, "X_train.npy"), X_train)
#np.save(os.path.join(save_dir, "X_test.npy"), X_test)
#np.save(os.path.join(save_dir, "y_train.npy"), y_train)
#np.save(os.path.join(save_dir, "y_test.npy"), y_test)

# Dimensions of the datasets
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)

xgb_model = XGBClassifier(
    n_estimators=100,     # number of boosted trees
    max_depth=5,          # tree depth (controls complexity)
    learning_rate=0.1,    # step size for each boosting step
    subsample=0.8,        # trains each tree on 80% of samples
    colsample_bytree=0.8, # use 80% of features per tree
    eval_metric="logloss" # required to suppress warnings for binary classification
)

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Predictions on test set
y_pred_xgb = xgb_model.predict(X_test) 

# Evaluate accuracy
xgb_acc = accuracy_score(y_test, y_pred_xgb)
print("XGBoost accuracy:", accuracy_score(y_test, y_pred_xgb))

# Support Vector Machine (SVM)

#Train a SVM model

# A linear kernel is fast and is commonly used as a baseline model
svm_model = SVC(kernel="linear") 

# Train the SVM on the training set (16 flattened images)
svm_model.fit(X_train, y_train) 

# Predict class labels on the test set (4 images)
y_pred_svm = svm_model.predict(X_test) 

# Print SVM accuracy
print("SVM accuracy:", accuracy_score(y_test, y_pred_svm))


# Deep Learning Models


