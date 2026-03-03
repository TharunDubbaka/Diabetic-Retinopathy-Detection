# Diabetic-Retinopathy-Detection
This is an efficient B0 model trained on retinal fundus images. It classifies the retina image into which stage or DR it is in.



## Diabetic Retinopathy Detection using EfficientNet and Grad-CAM

An end-to-end deep learning pipeline for automated Diabetic Retinopathy (DR) detection from retinal fundus images using transfer learning and explainable AI techniques.

This project classifies retinal images into five severity levels and provides visual explanations using Grad-CAM to highlight important regions influencing the model’s prediction.

## Problem Statement

Diabetic Retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected early. Manual diagnosis from retinal images requires medical expertise and is time-consuming.

This project builds an automated system that:

Classifies retinal images into five DR stages

Handles class imbalance effectively

Provides model interpretability through Grad-CAM visualization

## Model Architecture

The model is implemented using:

PyTorch

EfficientNet (EfficientNet-B0 pretrained on ImageNet)

Modifications

Replaced the final classifier layer to output five classes

Applied weighted CrossEntropyLoss to address class imbalance

Used Adam optimizer with learning rate 1e-4

## DR Classification Labels
Class Index	Label
0	No_DR
1	Mild
2	Moderate
3	Severe
4	Proliferative
Project Structure
├── train.py          # Model training script
├── test.py           # Inference + Grad-CAM visualization
├── output/
│   └── best_model.pth
├── train.csv
└── gaussian_filtered_images/
## Training Pipeline
Data Processing

Stratified train-validation split (85% / 15%)

Image resizing to 224×224

Data augmentation:

Random rotation

Horizontal and vertical flips

Color jitter

ImageNet normalization

Training Details

Batch size: 16

Epochs: 30

Loss function: Weighted CrossEntropyLoss

Optimizer: Adam

Best model saved based on validation accuracy

Training and validation accuracy and loss curves are plotted for performance monitoring.

Explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize the regions of the retinal image that most influence the model’s prediction.

## The inference pipeline:

Loads the trained EfficientNet-B0 model.

Preprocesses a new retinal image.

Predicts the DR stage.

Generates a heatmap overlay highlighting important regions.

This improves transparency and makes the system more suitable for medical decision-support applications.

## Technologies Used

Python

PyTorch

torchvision

NumPy

OpenCV

Matplotlib

scikit-learn

## Potential Applications

Automated DR screening systems

Clinical decision-support tools

AI-assisted ophthalmology diagnosis

Early detection in rural or resource-limited settings
