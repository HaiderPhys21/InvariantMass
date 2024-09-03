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

### Deep Learning Approaches

- **DNN (Deep Neural Network):** A feedforward neural network with multiple layers that processes tabular data directly.
- **CNN (Convolutional Neural Network):** A neural network particularly effective for image data, applied here by converting tabular data into image representations. The CNN's capability to learn spatial hierarchies within the data makes it a powerful tool for this task, though it introduces challenges in how the data is represented.

## Methodology

### Data Processing

The project involves several key data processing steps:

1. **Feature Extraction:**
   - The dataset contains multiple features for each electron in an event, including energy $$(\(E\))$$, momentum components $$(\(p_x\)$$, $$\(p_y\)$$, $$\(p_z\))$$, transverse momentum $$(\(p_t\))$$, pseudorapidity $$(\(\eta\))$$, azimuthal angle $$(\(\phi\))$$, and charge $$(\(q\))$$.
   - These features were normalized to ensure that the network could process them effectively.

2. **Image Encoding:**
   - For the CNN, the challenge was to convert the tabular data into a 2D image-like format. This was done by mapping the features onto different channels of the image. Specifically:
     - The first channel encoded the transverse momentum $$(\(p_t\))$$ and energy $$(\(E\))$$.
     - The second channel represented the spatial information through pseudorapidity $$(\(\eta\))$$ and azimuthal angle $$(\(\phi\))$$.
     - The third channel included the momentum components $$(\(p_x\)$$, $$\(p_y\)$$, $$\(p_z\)$$).
   - This encoding ensured that the CNN could exploit the spatial relationships between these physical quantities.

### Model Development

Two models were developed and trained:

1. **Deep Neural Network (DNN):**
   - The DNN directly processed the normalized tabular data. The architecture included several fully connected layers with ReLU activation functions, followed by a final output layer that predicted the invariant mass.

2. **Convolutional Neural Network (CNN):**
   - The CNN was designed to process the 2D image representation of the data. The architecture included convolutional layers that captured spatial hierarchies within the data, followed by pooling layers to reduce dimensionality, and fully connected layers leading to the output.

### Mathematical Details

- **Loss Function:** Both models were trained using the Mean Squared Error (MSE) loss function, which is defined as:

\[$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\$$]

where $$\(y_i\)$$ is the true invariant mass and $$\(\hat{y}_i\)$$ is the predicted invariant mass.

- **Optimization:** The models were optimized using the Adam optimizer, which adjusts the learning rate during training to converge more efficiently.

### Model Evaluation

The models were evaluated using:

- **Mean Absolute Error (MAE):** Measures the average magnitude of the errors without considering their direction.
- **R-squared $$(\(R^2\))$$ score:** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Dataset

The dataset contains 100,000 simulated dielectron events, each described by the following features for each electron:

- **Energy (E)**
- **Momentum components (px, py, pz)**
- **Transverse Momentum (pt)**
- **Pseudorapidity (η)**
- **Azimuthal Angle (φ)**
- **Charge (q)**

The target variable is the **invariant mass (M)** of the electron pair.

## Model Architectures

### Deep Neural Network (DNN)

- **Input Layer:** Accepts the tabular data directly.
- **Hidden Layers:** Several fully connected layers with ReLU activations.
- **Output Layer:** A single neuron outputting the predicted invariant mass.

### Convolutional Neural Network (CNN)

- **Input Layer:** Processes the 3-channel image representation of the data.
- **Convolutional Layers:** Extract features from the image through spatial hierarchies.
- **Pooling Layers:** Reduce the dimensionality of the feature maps.
- **Fully Connected Layers:** Integrate the extracted features to output the predicted invariant mass.

## Results

The study compared the predictive accuracy of the DNN and CNN models, with key findings:

- **Performance Metrics:** 
  - The DNN model demonstrated superior performance on tabular data, as expected for structured input.
  - The CNN model, while slightly less accurate, provided insights into how spatial relationships in the data could be leveraged, but also revealed potential information loss during data conversion.



- **Interpretation:** 
  - The DNN's performance underscores its suitability for structured data, while the CNN's results suggest that the representation of physical quantities as images must be carefully designed to avoid losing critical information.

## Technologies Used

- **Programming Languages:** Python
- **Libraries:** TensorFlow, Pandas, NumPy, Matplotlib, scikit-learn
- **Tools:** Jupyter Notebook

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



