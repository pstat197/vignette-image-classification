# Classical ML and Deep Learning for Image Classification

This is a vignette on the application of classical machine learning methods (Support Vector Machines and XGBoost) and deep learning methods (Convolutional Neural Networks and Vision Transformers) for cat-vs-dog image classification; created as a class project for PSTAT 197A Fall 2025.

Contributors: Alex Morifusa, Joy Chang, Cathy Fang, Anna Liang, Madhav Rao

## Abstract


## Convolutional Neural Network (CNN)

### Overview
We implemented an improved CNN architecture designed to capture hierarchical visual features. This model serves as our deep learning baseline, with a balance between computational efficiency and learning capacity.

### Model Features & Improvements
* CNNs rely on convolutional layers that progressively extract hierarchical features from images, starting from simple edges and textures to more complex shapes and object patterns. This layered, spatially local approach allows CNNs to perform exceptionally well on visual tasks, even with relatively small datasets. 

* Our improved CNN incorporated multiple convolutional and pooling layers, data augmentation, dropout for regularization, and early stopping to prevent overfitting. 

### Performance
* Training Accuracy: We trained the model for 5 epochs using a batch size of 32. The model showed healthy convergence, with training accuracy rising to 77.35% and validation accuracy reaching 79.14%. The fact that validation accuracy tracks closely with (or slightly exceeds) training accuracy suggests our regularization techniques prevented overfitting.

* Test Accuracy: When evaluated on the held-out test set of 5,000 images, the model achieved an overall accuracy of 79.90%.

* Class Balance: The model performs symmetrically, though slightly differently per class:

    * Cats: Higher Recall (0.83) — The model is better at finding cats and misses fewer of them.

    * Dogs: Higher Precision (0.82) — When the model predicts "Dog," it is more likely to be correct, though it misses some actual dogs (Recall 0.77).

* Summary: Achieving ~80% accuracy in just 5 epochs highlights the efficiency of CNN for image tasks, though a ~20% error rate suggests there is still room for improvement compared to state-of-the-art Vision Transformers.

## Vision Transformer (ViT)

* Based on Google’s paper **“An Image is Worth 16×16 Words”**, we selected the Vision Transformer (ViT) as one of our models.
* ViT uses neural networks like CNNs but relies on the **transformer architecture**, where **self-attention** allows the model to analyze an image holistically rather than through layered convolutions.

### Advantages
* When **pretrained**, ViT achieves performance comparable to CNNs while using **less compute**.
* Excels at capturing **long-range dependencies** and global relationships in images.

### Limitations
* Requires **much more training data** than CNNs when trained from scratch.
* In our experiments, a ViT trained from scratch reached **~66% accuracy**, while the **pretrained model reached ~97%**.

### Data Processing & Training Pipeline

#### Image Preprocessing
* Converted raw cat and dog image files into NumPy arrays using a custom preprocessing function.
* Standardized the images so they were ready for PyTorch input.

#### Train/Test Split
* Split the dataset into 80% training and 20% testing.
* Used stratified sampling to maintain balanced proportions of cats and dogs across both sets.

#### PyTorch Pipeline Setup
* Wrapped the processed images into custom PyTorch `Dataset` classes.
* Used `DataLoader` objects for efficient batching, shuffling, and parallel data loading.

#### Model Training
* Fine-tuned a pretrained Vision Transformer (ViT) model on the training dataset.
* Leveraged pretrained weights to achieve strong performance on a relatively small dataset.

#### Evaluation
* Evaluated the model on the test set and printed sample predictions with confidence scores.
* Demonstrated how the pretrained ViT benefits feature extraction even with limited training data.

#### Note
* Due to the large dataset size, the notebook kernel crashed repeatedly, preventing final accuracy results from being recorded (`data.ipynb`).


## References

https://arxiv.org/pdf/2010.11929

2 required, need 1 more

