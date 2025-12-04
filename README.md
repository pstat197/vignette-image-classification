# vignette-image-classification

## XGBoost and SVM baseline models for image classification

### Purpose of this branch
- Implement image preprocessing for classical ML models  
- Train and evaluate **XGBoost** and **SVM** models on flattened image data  
- Produce accuracy values and confusion matrices
    - Conclude that these models serve as baseline models and do not effectively capture spatial patterns (edges, textures, shapes) as effectively as CNNs or vision transformers
    - Segue into deep learning models

Note: 
The datasets are too large for github, so they are not included in this repository. 
Please download the dataset separately from the official source (e.g., Kaggle Cats vs Dogs).  
Place the images in a local `data/` directory before running the notebook.  
