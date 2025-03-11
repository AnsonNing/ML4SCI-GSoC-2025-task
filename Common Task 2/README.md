# Common Task 2: Jets as graphs

## Overview
This project is part of **Common Task 2**, where a **Graph Neural Network (GNN)** is used to classify quark/gluon jet events. The original dataset consists of **3-channel images** (ECAL, HCAL, and Tracks), which are converted into **point clouds** and further transformed into **graphs**. A **GraphSAGE** model is then trained to classify these jet events.

## Dataset
The dataset consists of **125x125 pixel images** with three channels:
- **ECAL (Electromagnetic Calorimeter)**
- **HCAL (Hadronic Calorimeter)**
- **Tracks**

Each sample represents a jet event, and the dataset is stored in an **HDF5 file**.

## Data Processing

### **1. Converting Images to Point Clouds**
- The dataset images are **masked**, and only non-zero pixel coordinates are extracted.
- Each extracted point contains the **feature values (ECAL, HCAL, Tracks)** along with **(x, y) coordinates**.

### **2. Constructing Graphs**
- A **K-Nearest Neighbors (KNN) graph** is built using **k=5**.
- The **edge list** is created based on spatial proximity, and each edge stores **Euclidean distance** as an edge feature.
- The dataset is stored as a `torch_geometric.data.Data` object.

### **3. Dataset Splitting**
- The dataset is **split into 80% training, 10% validation, and 10% test sets**.
- PyTorch Geometric's `DataLoader` is used for efficient batch processing.

## Model Architecture

The model is based on **GraphSAGE**, a Graph Neural Network (GNN) designed for **graph-based classification**.

### **GraphSAGE Model**
- **Graph Convolution Layers**:
  - 5 layers of `SAGEConv` for **message passing** and **feature extraction**.
  - Each layer applies **LayerNorm** and **ReLU activation**.
  - **Residual connections** are used for stable training.
- **Pooling & Classification**:
  - **Global Max Pooling** aggregates graph-level representations.
  - A **fully connected layer** outputs **2-class predictions** (quark vs. gluon).
- **Dropout** (0.3) is applied for **regularization**.

## Training Details

- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam (`lr=1e-3`, `weight_decay=5e-5`)  
- **Scheduler:** Cosine Annealing LR with Warm-up (`80 epochs`)  
- **Batch Size:** 32  
- **Number of Epochs:** 80  

## Training Process
- **Automatic Mixed Precision (AMP)** is used for **faster training**.
- **Gradient scaling** ensures stability in mixed precision training.
- The **best model (based on validation accuracy) is saved** automatically.

## Evaluation
- The trained model is evaluated on the **test set**.
- Metrics include **loss and classification accuracy**.
- The training process is visualized using **loss and accuracy curves**.

## Results

### **Training, Validation & Test Loss**
*(You can add a loss curve image here)*
![Loss Curve](path_to_your_loss_curve.png)

### **Training, Validation & Test Accuracy**
*(You can add an accuracy curve image here)*
![Accuracy Curve](path_to_your_accuracy_curve.png)

### **Final Test Accuracy**
- The best model is evaluated on the **test set**.
- **Final test accuracy: X.XXX%** *(Replace with your actual result)*.

