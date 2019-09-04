# Champs_GraphConV

This project is my implementation for using the **graph convolution neural network** for the Champs competition on Kaggle. This is showcase from data preprocessing to the final result. It is developed base on the torch-geometric library.

To preprocessing data, you would need to install two library such as dscribe and rdkit to be able to extract the feature about atom and molecules.

Installation for dependencies:
```
pip install networkx dscribe
conda install -y -c rdkit rdkit
```

In this experiment, serveral configuration and different neural network is experimented. It seems that adding the attention mechanism into network that has a significant impact on the performance rather than pure simple graph conv.
