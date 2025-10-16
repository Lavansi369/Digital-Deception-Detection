
1. Overview

This project presents a lightweight and explainable framework for detecting deepfake manipulations in static images. In an era where hyper-realistic synthetic media poses a significant threat to information integrity, this work moves beyond computationally expensive, end-to-end deep learning models to explore the efficacy of a feature-driven approach.

Our methodology is centered on multi-modal feature extraction, where we create a robust "fingerprint" for each image by combining two fundamentally different types of information:

Spatial & Geometric Features: Derived from facial landmarks to analyze the physical structure, proportions, and symmetry of the face.

Frequency-Domain Features: Derived from a Fast Fourier Transform (FFT) to identify subtle, pixel-level artifacts and textural inconsistencies common in generated images.

This hybrid feature set is then used to train a powerful classifier, demonstrating a high degree of accuracy in distinguishing real images from fakes.

2. Methodology Pipeline

The project is structured into three distinct, modular stages:

Stage 1: Feature Extraction

A custom Python script (generate_final_features.py) processes a dataset of real and fake images. For each image, it calculates a comprehensive 1D feature vector that includes:

Normalized Ratios: Scale-invariant measurements like eye-distance-to-face-height ratio.

"Expert" Features: Asymmetry scores and Eye/Mouth Aspect Ratios (EAR/MAR) to detect unnatural expressions.

Frequency Artifacts: The High-Frequency Energy Ratio (HFER) and variance from the FFT spectrum.

The output of this stage is a single CSV file containing the complete feature set for the entire dataset.

Stage 2: Model Training

The generated CSV file is used to train a classifier. This repository contains a script (train_model_independent.py) that:

Splits the data into training, validation, and test sets.

Applies StandardScaler to normalize the features.

Trains a high-performance XGBoost model.

Evaluates the model and generates key performance metrics (Accuracy, Precision, Recall, F1-Score) and a confusion matrix.

Saves the final "Prediction Package" (model, scaler, and feature_columns).

Stage 3: Prediction

A final script (predict_final.py) loads the trained "Prediction Package" and uses it to provide a "REAL" or "FAKE" verdict on new, unseen images that are not part of the original dataset.

3. How to Use This Project

Follow these steps to replicate the workflow:


Step 1: Prepare the Dataset

Download the "Real and Fake Face Detection" dataset from Kaggle (or your chosen source).

Create a dataset/ folder in the project root and organize the images with the following structure:

dataset/
├── training_real/
│   ├── real_image_001.jpg
│   └── ...
└── training_fake/
    ├── fake_image_001.jpg
    └── ...


Step 2: Generate the Feature Set

Run the feature extraction script. Make sure to update the dataset_path variable inside the script.

python generate_final_features.py


This will create a final_deepfake_features.csv file, which will be ignored by Git.

Step 3: Train the Model

Run the training script. It will automatically load the CSV file created in the previous step.

python train_model_independent.py


This will train the model, display the evaluation results, and save the deepfake_model.joblib, scaler.joblib, and feature_columns.joblib files.

Step 4: Test on New Images

Place your test images (e.g., test.jpg) in the root directory and run the prediction script.

python predict_final.py


This will display your test images with the final "REAL" or "FAKE" verdict.

4. Technology Stack

Core Language: Python 3.x

Feature Extraction: MediaPipe, OpenCV, NumPy

Data Handling: Pandas

Model Training & Evaluation: Scikit-learn, XGBoost

Visualization: Matplotlib, Seaborn
