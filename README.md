# Classification-on-sales-data
Predictive Analytics Project
Introduction
This project aims to predict the number of products sold by users based on various demographic and behavioral features. We employ machine learning techniques to analyze a dataset containing user information and product sales data.

Dataset
The dataset, located at /content/6M-0K-99K.users.dataset.public.csv, consists of user attributes such as age, gender, country, seniority, app usage, and social interactions.

Methodology
Data Preprocessing:

Explored the dataset's structure and handled missing values.
Analyzed the distribution of numerical features and visualized relationships between variables.
Encoded categorical features and converted boolean columns to integer type.
Exploratory Data Analysis (EDA):

Investigated relationships between features and the target variable (products sold).
Detected and handled outliers to improve model performance.
Model Building:

Split the dataset into training and testing sets.
Trained regression models (Linear Regression, Random Forest Regression, XGBoost Regression) to predict product sales.
Evaluated model performance using mean squared error (MSE) and R-squared (R2) score.
Trained a classification model (XGBoost Classifier) to predict product sales categories.
Evaluated classification model performance using accuracy, precision, recall, F1 score, and confusion matrix.
Model Selection and Hyperparameter Tuning:

Compared the performance of different classification algorithms (Logistic Regression, K-Nearest Neighbors, Support Vector Machine).
Performed hyperparameter tuning for selected models using GridSearchCV or manually specified hyperparameters.
Evaluated tuned model performance using various metrics.
Results
The Random Forest Regression model outperformed other regression models with the lowest MSE and highest R2 score.
For classification, the XGBoost Classifier achieved the highest accuracy and F1 score among the tested algorithms.
Hyperparameter tuning improved model performance, particularly for Logistic Regression and Support Vector Machine.
Conclusions
Demographic and behavioral features significantly impact product sales.
Regression models effectively predict the number of products sold based on user attributes.
Classification models can categorize product sales, aiding in customer segmentation and targeted marketing strategies.
Future Directions
Explore more advanced feature engineering techniques to improve model performance.
Incorporate additional data sources for a more comprehensive analysis.
Deploy models in real-world scenarios to automate product sales predictions and enhance business decision-making processes.
Dependencies
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
