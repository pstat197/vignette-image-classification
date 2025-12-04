# Classical ML and Deep Learning for Image Classification

This is a vignette on the application of classical machine learning methods (Support Vector Machines and XGBoost) and deep learning methods (Convolutional Neural Networks and Vision Transformers) for cat-vs-dog image classification; created as a class project for PSTAT 197A Fall 2025.

Contributors: Alex Morifusa, Joy Chang, Cathy Fang, Anna Liang, Madhav Rao

## Abstract

## Repository Contents
- comparison_vignette.ipynb (or .qmd)

    - The Main Tutorial Document. This is the primary file to read.

    - It contains all written explanations, imports functions from scripts/, and loads saved results from results/ to generate plots instantly.

- scripts/

    - The Code Repository. Contains the functional, reusable Python files (.py) for all three models.

- data/

    - The Dataset Location. Contains the raw image files (e.g., train/).

    - Note: This folder is included in .gitignore and is not committed to the repository due to file size limits. Users must run 01_download_data.py to populate it.

- data_sample/

    - Demo Dataset Location. This holds a very small set of images used specifically for quick demonstrations inside the main notebook.

- img/

    - Contains images generated during the analysis.


## References

https://arxiv.org/pdf/2010.11929

https://ieeexplore.ieee.org/abstract/document/9736894