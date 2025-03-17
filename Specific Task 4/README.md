# Specific Task 4: Non-Local GNNs for Jet Classification

## Overview
This project is part of **Specific Task 4**, where a **Non-Local Graph Neural Network (GNN)** is implemented to classify quark/gluon jet events. The model is compared against a **baseline GCN model** using **ROC-AUC** as the evaluation metric.

## Dataset
The dataset consists of 125x125 pixel images with three channels:
- **ECAL (Electromagnetic Calorimeter)**
- **HCAL (Hadronic Calorimeter)**
- **Tracks**

Each sample represents a jet event, and the dataset is stored in an HDF5 file (`quark-gluon_data-set_n139306.hdf5`).

## Data Processing

### 1. Splitting the Dataset
The dataset is processed into a graph structure by batching nodes and edges using `torch_geometric.data.Data`.  
Each graph sample corresponds to a jet event.

### 2. Normalizing Point Cloud Features
- The dataset undergoes global standardization using `StandardScaler()`.
- Each **ECAL, HCAL, and Tracks channel** is independently normalized to ensure consistency across different jet events.

### 3. Converting Images to Point Clouds
- The **non-zero pixels** in the dataset images are extracted as (x, y) coordinates.
- The extracted features are normalized using the scalers.
- The resulting point cloud representation preserves the spatial structure of the jet events.

### 4. Constructing Graphs
- A K-Nearest Neighbors (KNN) graph is built using `k=5`, where each node connects to its 5 nearest neighbors.
- The edge list is created based on spatial proximity.
- Each edge stores **Euclidean distance** and **feature differences** as attributes.

### 5. Processing the Graph Dataset
- Each image is converted into a graph with nodes representing features of active pixels.
- All graphs are batched together into a `torch_geometric.data.Data` object for efficient processing in PyTorch Geometric.

## Model Architectures

### 1. Baseline GCN Model
The **Graph Convolutional Network (GCN)** model consists of:
- **GCN Layers:** Two `GCNConv` layers for message passing.
- **ReLU Activation & Dropout:** Improve generalization.
- **Global Mean Pooling:** Aggregates node features into a graph-level representation.
- **Fully Connected Layer:** Produces final class predictions.

### 2. Hybrid Non-Local GNN Model
A **Hybrid GATv2-GCN Model** is implemented with:
- **GATv2 Layers:** Capture attention-based non-local dependencies.
- **GCN Layer:** Combines local spatial features with global attention.
- **Global Mean Pooling:** Aggregates learned graph features.
- **Fully Connected Layer:** Outputs the classification prediction.

## Training Details

- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam (`lr=1e-3`, `weight_decay=5e-5`)  
- **Scheduler:** Cosine Annealing LR (`50 epochs`)  
- **Gradient Scaling:** Automatic Mixed Precision (AMP) for stability  
- **Batch Size:** 32  

## Training Process
- Both GCN and Hybrid GATv2-GCN models are trained simultaneously.
- Gradient scaling with AMP ensures stability in mixed precision training.
- The Cosine Annealing Scheduler adjusts the learning rate dynamically.

## Results

### ROC-AUC Comparison
![Training ROC-AUC](curve.png)

### Final Test ROC-AUC Scores
| Model | ROC-AUC (Test) |
|------------|---------------|
| **Baseline GCN** | 0.000 | 
| **Hybrid GATv2-GCN** | 0.000 | 


