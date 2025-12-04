# vignette-image-classification

## XGBoost and SVM baseline models

The preprocessing script contains two functions. The first function preprocesses the data to be used for classical models such as XGBoost and Support Vector Machine.

The function prepare_ml_data Loads images, resizes them to 64×64, normalizes pixel values, and flattens each image into a 1D feature vector, producing a 2D feature matrix X.

