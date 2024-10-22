

# Customer Churn Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Technologies Used](#technologies-used)
5. [Model Performance](#model-performance)
6. [Installation and Setup](#installation-and-setup)
7. [How to Use](#how-to-use)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction

The **Customer Churn Prediction** project aims to identify customers who are likely to stop using a service or product, known as "churning." Predicting churn allows businesses to take proactive steps to retain customers, thereby improving customer satisfaction and revenue. This project utilizes machine learning techniques to classify customers based on whether they are likely to churn or remain.

## Key Features

- Predicts whether a customer is likely to churn using historical data.
- Analyzes factors contributing to customer churn such as:
  - Contract length
  - Payment methods
  - Monthly charges
  - Internet services
- Uses various machine learning models for classification, including:
  - Logistic Regression
  - Random Forest
  - Decision Trees
  - Support Vector Machines (SVM)
  - Gradient Boosting
- Provides data visualization to better understand churn patterns.

## Dataset

The dataset used in this project contains features that describe customer behavior and demographics, typically including:
- **Customer ID**: Unique identifier for each customer.
- **Gender**: Customer's gender (Male/Female).
- **Senior Citizen**: Whether the customer is a senior citizen (Yes/No).
- **Partner**: Whether the customer has a partner (Yes/No).
- **Dependents**: Whether the customer has dependents (Yes/No).
- **Tenure**: Number of months the customer has stayed with the company.
- **Phone Service**: Whether the customer has phone service (Yes/No).
- **Multiple Lines**: Whether the customer has multiple lines (Yes/No).
- **Internet Service**: Type of internet service the customer has (DSL, Fiber Optic, None).
- **Contract**: The type of contract the customer has (Month-to-Month, One Year, Two Year).
- **Monthly Charges**: The amount the customer is billed monthly.
- **Total Charges**: Total charges incurred by the customer.
- **Churn**: Target variable (Yes/No).

### Data Preprocessing:
- **Handling missing values**: Removed or imputed any missing data.
- **Feature encoding**: Converted categorical variables into numerical format using techniques like one-hot encoding.
- **Feature scaling**: Standardized numerical features such as monthly charges and tenure to ensure better model performance.

## Technologies Used

- **Jupyter Notebook**: Development environment for writing and running the code.
- **Python**: Programming language.
- **Pandas and NumPy**: For data manipulation and preprocessing.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib and Seaborn**: For data visualization and analysis.

## Model Performance

Various machine learning models were trained and evaluated to predict customer churn. The models were compared based on performance metrics such as accuracy, precision, recall, and F1-score.

### Example Models:
- **Logistic Regression**: Baseline model for binary classification.
- **Random Forest**: An ensemble learning method that works well with large datasets.
- **Gradient Boosting**: A powerful boosting method for improving classification accuracy.

### Evaluation Metrics:
- **Accuracy**: The percentage of correct predictions (e.g., 80% accuracy).
- **Precision**: Proportion of correctly predicted churn customers to the total predicted churn customers.
- **Recall**: Proportion of actual churn customers that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, used to balance between the two metrics.

For example:
- Logistic Regression achieved an accuracy of **78%**.
- Random Forest achieved an accuracy of **85%** and provided a more balanced F1-score compared to other models.

## Installation and Setup

To run this project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies**:
   Use `pip` to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Run the notebook**:
   Open the `Customer_Churn_Prediction.ipynb` file in Jupyter Notebook and run the cells to preprocess the data, train the model, and evaluate the results.

## How to Use

1. **Preprocess the data**:
   The notebook includes steps for cleaning, encoding, and transforming the dataset into a suitable format for modeling.

2. **Train the model**:
   Several models are provided in the notebook. You can modify hyperparameters and retrain the models to optimize their performance.

3. **Evaluate the model**:
   Use evaluation metrics like accuracy, precision, recall, and F1-score to determine the model's performance on the test dataset.

4. **Predict new churn data**:
   Once the model is trained, you can use it to predict whether new customer data will result in churn or not.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

