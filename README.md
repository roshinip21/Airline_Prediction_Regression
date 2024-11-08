# Airline Price Prediction Using Regression

This project aims to predict airline ticket prices based on various factors such as departure time, destination, airline, and more. It uses multiple machine learning models to compare and determine the best approach for accurate price prediction. The project is structured according to the CRISP-DM framework, guiding the process from data understanding to model deployment and evaluation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Technologies Used](#technologies-used)
4. [Project Workflow](#project-workflow)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

In the airline industry, pricing is highly dynamic and can fluctuate based on a wide range of factors, including time of year, route, airline, and more. The objective of this project is to develop a regression model that can predict airline ticket prices, helping users understand the cost trends and potentially plan travel more efficiently.

## Data

This project uses a dataset containing airline ticket price information. Each row includes details like:
- **Airline**: Name of the airline.
- **Date of Journey**: Date when the ticket was booked.
- **Source**: Starting location.
- **Destination**: Destination location.
- **Departure and Arrival Times**: Scheduled times for departure and arrival.
- **Duration**: Flight duration.
- **Price**: Target variable representing ticket price.

The dataset is preprocessed, cleaned, and split for training and testing the models.

## Technologies Used

The project utilizes the following key technologies and libraries:

- **Python**: The core language for the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib and Seaborn**: For data visualization and exploratory data analysis.
- **Scikit-learn**: For building and evaluating machine learning models.
- **termcolor**: To enhance console print statements.

### Key Libraries and Tools

| Library       | Purpose                                |
|---------------|----------------------------------------|
| `pandas`      | Data handling and manipulation         |
| `numpy`       | Numerical operations                   |
| `matplotlib`  | Basic plotting and visualization       |
| `seaborn`     | Enhanced statistical data visualization|
| `termcolor`   | Colored text in terminal output        |
| `scikit-learn`| Machine learning modeling and evaluation|

## Project Workflow

This project follows the CRISP-DM framework to ensure a structured approach:

1. **Business and Data Understanding**: Define the problem scope and gather requirements for the data. Load and explore the data for insights.
2. **Data Engineering & Preparation**: Clean, preprocess, and transform the data to prepare for modeling.
3. **ML Model Engineering**: Build multiple regression models, including:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
4. **Quality Assurance for Machine Learning Applications - Model Evaluation**: Evaluate models based on metrics like Mean Absolute Error, Mean Squared Error, and R-squared to select the best model.
5. **Model Deployment, Monitoring & Maintenance**: Although this step is a placeholder, in a production environment, we could use platforms like AWS Sagemaker or Azure ML for deployment and monitoring.

## Installation

To set up the project locally, please follow these instructions:

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Required Libraries**:

   Install the libraries listed in the `requirements.txt` file using:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install each library individually:

   ```bash
   pip install pandas numpy matplotlib seaborn termcolor scikit-learn
   ```

## Usage

To use the project:

1. **Open the Jupyter Notebook**:

   Open `airlineprice_regression.ipynb` using Jupyter Notebook or JupyterLab.

   ```bash
   jupyter notebook airlineprice_regression.ipynb
   ```

2. **Run the Code Cells**:

   The notebook is structured sequentially; running each cell step-by-step will execute the data loading, preprocessing, modeling, and evaluation stages.

3. **Adjust Hyperparameters**:

   For experimentation, you may alter the hyperparameters for each model within the notebook to see how they affect model performance.

4. **View Results**:

   Outputs are provided in the notebook to interpret the model results and visualize predictions.

## Results

The project compares multiple models and provides an evaluation summary in terms of performance metrics:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

The best model selection depends on the lowest error and highest R-squared score, indicating a good fit for the data.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Open a pull request.

