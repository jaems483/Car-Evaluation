Car Evaluation Analysis Project
Project Overview

This project analyzes the Car Evaluation Dataset from the UCI Machine Learning Repository using Python and machine learning techniques. The objective of the analysis is to preprocess categorical vehicle evaluation data, explore feature relationships, build predictive classification models, optimize model performance, and generate predictions for new car configurations.

The project demonstrates a complete machine learning workflow including:

Data loading
Data preprocessing
Exploratory Data Analysis (EDA)
Feature encoding
Model training
Model evaluation
Hyperparameter tuning
Prediction on new unseen data

The analysis was implemented in Python using libraries such as Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, and Seaborn.

Dataset Information
Dataset Citation

Car Evaluation Dataset

Bohanec, M. (1988). Car Evaluation [Dataset]. UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5JP48

Dataset Description

The Car Evaluation dataset was originally developed to evaluate cars according to several categorical attributes.

Features
Feature	Description
Buying	Buying price of the car
maint	Maintenance cost
doors	Number of doors
persons	Passenger capacity
lug_boot	Luggage boot size
safety	Safety rating
class	Car acceptability classification (target variable)

Project Workflow
1. Data Loading

The dataset was imported into a Pandas DataFrame using:

pandas.read_csv()
Custom column naming
Processes Performed
Imported dataset from car.data
Assigned readable column names
Loaded the data into a structured DataFrame for analysis

Example:

cols = ['Buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv("car.data", names=cols)
2. Data Preprocessing

Since the dataset contains categorical values, ordinal encoding was applied to convert categories into numerical representations suitable for machine learning models.

Encoding Strategy
Buying & Maintenance Cost Mapping
{'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
Doors & Persons Mapping
{'2': 2, '3': 3, '4': 4, '5more': 5, 'more': 5}
Luggage Boot Size Mapping
{'small': 1, 'med': 2, 'big': 3}
Safety Mapping
{'low': 1, 'med': 2, 'high': 3}
Target Class Mapping
{'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
Processes Performed
Converted categorical variables into numerical format
Applied ordinal encoding based on feature hierarchy
Encoded the target variable for classification
Split the dataset into:
Features (X)
Target variable (y)
3. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to understand feature relationships and data patterns.

Correlation Analysis

A correlation heatmap was generated using Seaborn and Matplotlib to visualize relationships between variables.

Processes Performed
Generated feature correlation matrix
Visualized correlations using a heatmap
Identified potential relationships between features and target class

Example:

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
4. Data Splitting

The dataset was divided into training and testing subsets using Scikit-learn.

Processes Performed
Applied train_test_split()
Used:
80% training data
20% testing data
Ensured reproducibility with random_state=42

Example:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
5. Machine Learning Models

Two classification models were implemented and compared.

A. XGBoost Random Forest Classifier

The first model used was:

XGBoost

Processes Performed
Initialized XGBRFClassifier
Trained the model using training data
Generated predictions on the test set

Example:

model = xgb.XGBRFClassifier(random_state=42)
model.fit(X_train, y_train)
B. Neural Network Model

A Multi-Layer Perceptron (MLP) Neural Network was also implemented using:

Scikit-learn

Model Architecture
Hidden layers: (100, 50)
Activation function: ReLU
Optimizer: Adam
Maximum iterations: 500
Processes Performed
Configured neural network architecture
Trained the MLP classifier
Generated predictions for evaluation

Example:

MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500
)
6. Model Evaluation

Both models were evaluated using multiple classification metrics.

Evaluation Metrics Used
Metric	Purpose
Accuracy	Overall prediction correctness
Precision	Positive prediction quality
Recall	Ability to detect correct classes
F1 Score	Balance between precision and recall
Confusion Matrix	Detailed prediction comparison
Processes Performed
Calculated weighted classification metrics
Compared model performance
Printed confusion matrices for detailed analysis

Example:

accuracy_score()
precision_score()
recall_score()
f1_score()
confusion_matrix()
7. Hyperparameter Tuning

Grid Search Cross Validation was applied to optimize the XGBoost classifier.

Optimization Technique
GridSearchCV
5-fold cross-validation
Weighted F1 score optimization
Parameters Tuned
Parameter	Values Tested
n_estimators	100, 200, 300
max_depth	3, 5, 7
learning_rate	0.01, 0.1, 0.2
colsample_bytree	0.7, 0.9, 1.0
subsample	0.7, 0.9, 1.0
Processes Performed
Defined parameter grid
Performed exhaustive hyperparameter search
Selected the best-performing model
Evaluated optimized classifier on test data
8. Prediction on New Data

The optimized XGBoost model was used to predict the classification of a new car configuration.

Processes Performed
Created a new sample input
Applied trained model prediction
Converted numerical predictions back to categorical labels
Searched for similar entries in the original dataset

Example Prediction Workflow:

new_prediction = best_xgb_model.predict(new_data)
Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Key Learning Outcomes

This project demonstrates:

Categorical data preprocessing
Ordinal encoding techniques
Exploratory data analysis
Classification model development
Neural network implementation
Ensemble learning with XGBoost
Hyperparameter optimization
Model performance evaluation
Prediction pipeline development
Suggested Improvements

Future enhancements may include:

Additional feature engineering
Model explainability using SHAP values
Cross-model benchmarking
Automated preprocessing pipelines
Deployment using Flask or Streamlit
Model persistence using Pickle or Joblib
Conclusion

This project successfully applies machine learning techniques to classify car evaluations using categorical vehicle attributes. Multiple models were trained and evaluated, with hyperparameter tuning improving predictive performance.

The workflow demonstrates a complete end-to-end machine learning pipeline from preprocessing and EDA to optimization and prediction, making it a strong example of practical supervised learning implementation.

Author

Project: Car Evaluation Machine Learning Analysis
Dataset: Car Evaluation Dataset
Tools: Python, Scikit-learn, XGBoost, Pandas, Seaborn
