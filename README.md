
DIGITAL DECEPTION DETECTION

1. This project introduces a lightweight and interpretable framework designed for the detection of deepfake manipulations in static images. In a time when hyper-realistic synthetic media presents a considerable risk to the integrity of information, this end-to-end deep learning models to investigate the effectiveness of a feature-driven methodology.
Our approach focuses on multi-modal feature extraction, wherein we develop a robust "fingerprint" for each image by integrating two fundamentally distinct types of information:

 Spatial & Geometric Features: These are derived from facial landmarks to assess the physical structure, proportions, and symmetry of the face.

Frequency-Domain Features: These are obtained through a Fast Fourier Transform (FFT) to detect subtle pixel-level artifacts and textural inconsistencies that are typically found in generated images.

This combined feature set is subsequently utilized to train a powerful classifier, which exhibits a high level of accuracy in differentiating real images from fakes.

2. Methodology Pipeline

The project is organized into three separate, modular phases:

Stage 1: Feature Extraction

A custom Python script (generate_final_features.py) processes a dataset comprising both real and fake images. For each image, it computes a comprehensive 1D feature vector that encompasses:

Normalized Ratios: Scale-invariant metrics such as the eye-distance-to-face-height ratio.

"Expert" Features: Asymmetry scores and Eye/Mouth Aspect Ratios (EAR/MAR) to identify unnatural expressions.

Frequency Artifacts: The High-Frequency Energy Ratio (HFER) and variance derived from the FFT spectrum.

The result of this stage is a single CSV file that contains the complete feature set for the entire dataset.

Stage 2: Model Training

The generated CSV file is employed to train a classifier. This repository includes a script (train_model_independent.py) that:

Divides the data into training, validation, and test sets.

Utilizes StandardScaler to normalize the features.

Trains a high-performance XGBoost model.

Assesses the model and produces key performance metrics.
