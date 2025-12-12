# Classical ML and Deep Learning for Image Classification

This is a vignette on the application of classical machine learning methods (Support Vector Machines and XGBoost) and deep learning methods (Convolutional Neural Networks and Vision Transformers) for cat vs dog image classification; created as a class project for PSTAT 197A Fall 2025.

Contributors: Alex Morifusa, Joy Chang, Cathy Fang, Anna Liang, Madhav Rao

## Abstract
This vignette explores the progression from classical machine learning to deep learning for image classification using the Cats vs Dogs dataset from Kaggle. We start by building baseline models—SVM and XGBoost—trained on flattened image vectors. These baselines help show the limitations of classical ML when images are reduced to tabular form. We then introduce a Convolutional Neural Network (CNN), which uses the spatial structure of images, and briefly discuss Vision Transformers (ViT), which apply transformer architectures to sequences of image patches. Together, these models illustrate the evolution from traditional approaches to modern architectures, highlighting both the strengths and weaknesses of each in image classification tasks.


## Repository Contents
- data_sample/

    - Demo Dataset Location. This holds a very small set of images used specifically for quick demonstrations inside the main notebook.

- data/

    - The Dataset Location. Contains the raw image files (e.g., train/).

    - Note: This folder is included in .gitignore and is not committed to the repository due to file size limits.

- notebooks/

    - A directory reserved for draft notebooks.

- scripts/

    - The Code Repository. Contains the functional, reusable Python files (.py) for the preprocessing functions and all three models.

- vignette.ipynb

    - The Main Tutorial Document. This is the primary file to read.

    - It contains all written explanations and runnable codes for complete demonstrations and analysis of results


## References

https://arxiv.org/pdf/2010.11929

https://ieeexplore.ieee.org/abstract/document/9736894

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)