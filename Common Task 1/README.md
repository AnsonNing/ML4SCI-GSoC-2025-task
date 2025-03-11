# Common Task 1: Autoencoder for Quark/Gluon Jet Events

## Overview
This project implements an autoencoder using a U-Net architecture to learn representations of quark/gluon jet events from a dataset containing 3-channel images (ECAL, HCAL, and Tracks). The autoencoder is trained to reconstruct input events, allowing for anomaly detection or feature extraction.

## Dataset
The dataset consists of 125x125 images across three channels:
- **ECAL (Electromagnetic Calorimeter)**
- **HCAL (Hadronic Calorimeter)**
- **Tracks**

## Data Processing
1. **Dataset**  
   The dataset is stored in an HDF5 file (`quark-gluon_data-set_n139306.hdf5`) with:
   - `X_jets`: 3-channel images of quark/gluon jets.
   - `y`: Corresponding labels.

2. **Preprocessing**  
   - Convert images to PyTorch tensors.
   - Normalize images using dataset-wide mean and standard deviation.
   - Transpose the image dimensions from (125, 125, 3) to (3, 125, 125).
   - Split the dataset into an 80% training set and 20% testing set.

## Model Architecture
The autoencoder is based on a **U-Net** design with symmetric encoder-decoder layers. The model consists of:

- **Encoder:**
  - 5 convolutional layers with stride 2 for downsampling.
  - Batch normalization and ReLU activation for stability.
  - Feature map reduction from 3 channels → 64 → 32 → 16 → 8 → 4.

- **Decoder:**
  - 5 transposed convolutional layers for upsampling.
  - Skip connections to retain spatial information.
  - Additional convolution layers after concatenation to refine features.

- **Final Output:**
  - The output passes through interpolation to match the original 125x125 image resolution.

## Training Details
- **Loss Function:** L1 Loss (Mean Absolute Error)  
- **Optimizer:** AdamW (learning rate `1e-4`, weight decay `1e-4`)  
- **Learning Rate Scheduler:** Cosine Annealing LR (min learning rate `5e-6`, max epoch `30`)  
- **Batch Size:** 128  
- **Number of Epochs:** 30  

## Results: Side-by-Side Comparison of Original and Reconstructed Events
Below are visual comparisons of original and reconstructed jet events:

**Original vs. Reconstructed Events**  
*(You can add your images here)*  
![Original vs. Reconstructed](path_to_your_image.png)


