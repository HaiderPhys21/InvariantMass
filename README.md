# Predicting Invariant Mass of Dielectron Events Using Deep Learning

This project presents a comparative study of using Deep Neural Networks (DNNs) and Convolutional Neural Networks (CNNs) to predict the invariant mass (M) of dielectron events in high-energy physics.

## Table of Contents

- [Introduction](#introduction)
- [Background](#background)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The primary objective of this project is to explore the effectiveness of different deep learning models in predicting the invariant mass of dielectron events produced in particle collisions. This task is fundamental in high-energy physics for identifying and studying particles produced in such events.

## Background

### Invariant Mass in Particle Physics

Invariant mass is a critical quantity in particle physics that remains constant regardless of the reference frame. It is used to identify particles produced in high-energy collisions, such as those in the Large Hadron Collider (LHC).

### Deep Learning Approaches

- **DNN (Deep Neural Network):** A feedforward neural network with multiple layers that processes tabular data directly.
- **CNN (Convolutional Neural Network):** A neural network particularly effective for image data, applied here by converting tabular data into image representations.

## Dataset

The dataset contains 100,000 simulated dielectron events with the following features for each electron:

- **Energy (E)**
- **Momentum components (px, py, pz)**
- **Transverse Momentum (pt)**
- **Pseudorapidity (η)**
- **Azimuthal Angle (φ)**
- **Charge (q)**

The target variable is the **invariant mass (M)** of the electron pair.

## Model Architectures

### Deep Neural Network (DNN)

The DNN model is designed to handle tabular data directly, utilizing fully connected layers to predict the invariant mass.

### Convolutional Neural Network (CNN)

The CNN model is employed by converting the tabular data into a grid-like image structure, allowing the model to capture spatial relationships within the data.

## Results

The results of the study show a comparison between the predictive accuracy of the DNN and CNN models. Key findings include:

- Performance metrics such as Mean Squared Error (MSE), R-squared (R²), and model loss during training and validation.
- Insights into the suitability of CNNs for tasks traditionally handled by DNNs in particle physics.

## Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Pandas
- Numpy

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/HaiderPhys21/InvariantMass.git
    cd dielectron-invariant-mass-prediction
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:

    ```bash
    jupyter notebook electroncollision.ipynb
    ```

### Running the Models

You can run the models directly from the Jupyter notebook or convert the notebook into a Python script and run it:

```bash
jupyter nbconvert --to script electroncollision.ipynb
python electroncollision.py
