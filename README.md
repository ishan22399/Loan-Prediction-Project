# Loan Prediction Project

The **Loan Prediction Project** is a machine learning initiative designed to predict individuals' loan eligibility based on key factors. By leveraging the K-Nearest Neighbors (KNN) algorithm, enhanced with a custom distance function, this project delivers accurate insights into loan approval likelihood using historical data.

## Features

- **Data Analysis**: Perform in-depth exploratory data analysis (EDA) to uncover patterns and relationships in the dataset.  
- **Model Training**: Train predictive models using the KNN algorithm enhanced with a novel custom distance metric.  
- **Custom Distance Function**: Replace traditional metrics like Euclidean distance with the custom formula:  

```math
f(x_1, x_2) = 
\begin{cases} 
0 & \text{if } \sum(x_{1\_clean} - x_{2\_clean})^4 = 0 \\ 
10 \cdot \left[ \frac{\ln(e) \cdot \left(\ln \left| \sum(x_{1\_clean} - x_{2\_clean})^4 \right| - \ln \left| \sum(x_{1\_clean} - x_{2\_clean})^2 \right| \right)}{2 \cdot \ln(10)} \right] & \text{otherwise}
\end{cases}
```
- **Evaluation**: Assess the performance of trained models using appropriate metrics.
- **Deployment**: Save trained models for future use and provide a script for seamless loan eligibility prediction.

## Structure

- **Data**: Contains datasets for training and testing.
- **Scripts**: Python scripts for model training, prediction, and evaluation.
- **Models**: Saved models.
- **README.md**: Overview, installation, and usage guidelines.
- **requirements.txt**: List of dependencies.

## Usage

1. **Installation**: Clone the repository and install required packages.
2. **Training**: Run data preprocessing and model training script.
3. **Prediction**: Predict loan eligibility using the trained model on new data.

## Reference
For an in-depth understanding of the methodology and the custom distance function used in this project, please refer to the research paper: [Loan Prediction Using KNN with a Custom Distance Metric](https://doi.org/10.22541/essoar.172019478.88222627/v1).

## Contribution

Contributions are welcome! Submit pull requests for suggestions, improvements, or bug fixes.
