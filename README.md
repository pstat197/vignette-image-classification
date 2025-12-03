# vignette-image-classification


## Vision Transformer

Based on the paper, https://arxiv.org/pdf/2010.11929, by Google one of the models we decdied to user was the vision transformer. This model uses neural networks just like in CNN, but uses different overall architecture, namley the transformer architecture. In the transformers architecture, the attention mechanism allows for the model to look at an image holistically, rather than in layers as done in the CNNs. The main advantage this model offers is when pre-trained it requires less computation power to achieve the same results as CNNs. However, ViT requries far more training data compared to CNNs. Creating a new ViT from scratch had around 66% accuracy for training vs using a pre-trained model has around 97%. 