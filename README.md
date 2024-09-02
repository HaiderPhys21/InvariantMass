# Predicting Invariant Mass of Dielectron Events Using Deep Learning: A Comparative Study of DNN and CNN Approaches

![image](https://github.com/user-attachments/assets/265a4d3c-a2a0-48a4-b159-b2f818cdbe68)

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

Invariant mass is a critical quantity in particle physics, defined as the mass of a system of particles that remains constant regardless of the reference frame in which it is measured. For two particles, the invariant mass \(M\) is given by the relation:

$$
M^2 = (E_1 + E_2)^2 - (\mathbf{p}_1 + \mathbf{p}_2)^2
$$

where $$\(E_1\)$$ and $$\(E_2\)$$ are the energies of the two particles, and $$\(\mathbf{p}_1\)$$ and $$\(\mathbf{p}_2\)$$ are their respective momentum vectors. This quantity is particularly important in high-energy physics as it allows physicists to identify particles produced in collisions, such as those at the Large Hadron Collider (LHC), by analyzing the decay products of those particles.


For instance, when a heavy particle decays into two electrons, the invariant mass of the electron pair can be used to infer the mass of the original particle. This method is essential for discovering new particles and studying the properties of known ones.

### Mathematical and Physical Considerations

The invariant mass formula above is derived from the principles of special relativity and the conservation of energy and momentum. The challenge in particle physics experiments is that the data collected from detectors is often noisy and sparse. Therefore, accurately reconstructing the invariant mass from experimental data requires sophisticated data analysis techniques.

### Challenges in Data Representation

In this project, we explore the use of deep learning models to predict the invariant mass from simulated collision data. The data is typically presented in a tabular format, where each event (collision) is characterized by several features, including the energies and momenta of the particles involved.

One key challenge is how to represent this data for input into different types of neural networks. While Deep Neural Networks (DNNs) can directly process tabular data, Convolutional Neural Networks (CNNs) require the data to be converted into a grid-like image format. This conversion process can lead to potential "information loss" because the spatial relationships imposed by the image grid may not naturally reflect the true physical relationships present in the data. This project investigates these issues by comparing the performance of DNNs and CNNs in predicting the invariant mass.

### Deep Learning Approaches

- **DNN (Deep Neural Network):** A feedforward neural network with multiple layers that processes tabular data directly. DNNs are well-suited for tasks where the input data is structured and does not require spatial hierarchies to be learned.
- **CNN (Convolutional Neural Network):** A neural network particularly effective for image data, applied here by converting tabular data into image representations. CNNs are typically used when the data has a spatial structure, as they are designed to capture local patterns in the data.

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

The DNN model is designed to handle tabular data directly, utilizing fully connected layers to predict the invariant mass. It takes the input features as a vector and processes them through multiple layers to output a prediction for the invariant mass.

### Convolutional Neural Network (CNN)

The CNN model is employed by converting the tabular data into a grid-like image structure, allowing the model to capture spatial relationships within the data. This approach leverages the ability of CNNs to learn hierarchical patterns in data, although it comes with the challenge of ensuring that the spatial structure imposed by the grid aligns with the physical relationships between features.

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
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any feature request or bug fix.

## License

This project is licensed under the [MIT License](LICENSE).

---

By [Syed Haider Ali](https://github.com/HaiderPhys21)

---
