# vignette-image-classification


## Vision Transformer

Based on the paper, https://arxiv.org/pdf/2010.11929, by Google one of the models we decdied to user was the vision transformer. This model uses neural networks just like in CNN, but uses different overall architecture, namley the transformer architecture. In the transformers architecture, the attention mechanism allows for the model to look at an image holistically, rather than in layers as done in the CNNs. The main advantage this model offers is when pre-trained it requires less computation power to achieve the same results as CNNs. However, ViT requries far more training data compared to CNNs. Creating a new ViT from scratch had around 66% accuracy for training vs using a pre-trained model has around 97%. 

I preprocessed the cat and dog images using a custom function to convert them into arrays suitable for PyTorch. I then split the dataset into training and testing sets with an 80/20 ratio, ensuring stratification so that both sets contained proportional numbers of cats and dogs. These sets were wrapped into custom PyTorch Dataset and DataLoader objects to enable efficient batching and shuffling during training. Finally, I fine-tuned a pretrained ViT model on the training set and evaluated it on the test set, printing sample predictions with confidence scores, demonstrating how the model leverages pretrained features for high accuracy on a smaller dataset. Unfortunately, due to the large data size, my Kernal kept on crashing before I could get the final accuracy results (data.ipynb).
