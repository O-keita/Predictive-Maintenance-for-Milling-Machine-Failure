# Predictive Maintenance for Milling Machine Failure using XGBoost

## Objective
This project aims to predict machine failures in a milling process based on sensor data, process parameters, and product quality using XGBoost. By building a model that predicts when equipment failure is likely to occur, the goal is to minimize downtime and optimize maintenance scheduling.

## Project Steps

### 1. Problem Definition
- **Goal**: Predict whether a machine will fail during a production cycle using operational and sensor data.
- **Dataset**: A synthetic milling machine dataset with 10,000 data points and 14 features, including failure labels.

### 2. Data Exploration & Understanding
- **Features**: Explore each feature to understand its role in machine failures.
  - `UID`: Unique identifier for each data point.
  - `Product ID/Type`: Product quality variants (L, M, H) that may influence failure rates.
  - `Air & Process Temperature`: Continuous variables that reflect operational conditions, potentially affecting heat dissipation.
  - `Rotational Speed` and `Torque`: Mechanical parameters that impact power and strain on the machine.
  - `Tool Wear`: Time the tool has been used before replacement or failure.
  - `Machine Failure`: Target variable indicating if a failure occurred.
  - `Failure Modes`: Investigate failure modes (TWF, HDF, PWF, OSF, RNF) to understand failure conditions and frequencies.

#### Initial Analysis:
- Compute correlation matrix to check relationships between features and failure modes.
- Visualize distributions of key features (e.g., tool wear, air temperature, torque) for both failed and non-failed instances.
- Analyze the impact of product quality type (L, M, H) on failure likelihood.

### 3. Data Preprocessing
- **Handle Missing Data**: Ensure no missing values.
- **Encoding Categorical Variables**:
  - Encode `Product Type` (L, M, H) using techniques like one-hot encoding.
- **Feature Engineering**:
  - Create interaction features (e.g., `power = rotational speed * torque`).
  - Calculate `temperature_diff = process_temp - air_temp`.
  - Aggregate `tool wear` into thresholds or time windows.

### 4. Model Development with XGBoost
- **Target**: The `machine failure` label (binary classification: 0 = no failure, 1 = failure).
- **Model Choice**: XGBoost for its effectiveness with structured data and feature importance insights.
- **Train-Test Split**: Split the dataset into training and testing sets (e.g., 80-20 split).
- **Hyperparameter Tuning**: Optimize parameters such as:
  - `n_estimators`: Number of trees.
  - `learning_rate`: Shrinks the contribution of each tree.
  - `max_depth`: Limits tree depth to prevent overfitting.
  - `subsample`: Controls fraction of samples used for fitting trees.
  - `colsample_bytree`: Controls number of features per tree.

### 5. Model Training
- Train the XGBoost model on the training set.
- Evaluate performance on the test set using cross-validation to ensure generalization.

### 6. Model Evaluation
- **Evaluation Metrics**:
  - Accuracy
  - Precision & Recall (important for imbalanced datasets).
  - F1-Score (balances precision and recall).
  - ROC-AUC (measures ability to distinguish between classes).
- **Feature Importance**: Use XGBoost's feature importance metrics to identify critical factors in predicting failures (e.g., torque, tool wear).
- **Advanced Evaluation**:
  - Plot confusion matrix to understand the tradeoff between false positives and negatives.
  - Use SHAP (SHapley Additive exPlanations) values to explain individual predictions.

### 7. Visualization
- **Feature Importance Plot**: Visualize the most important features.
- **Failure Probability Curves**: Plot failure probabilities over time for different product variants.
- **SHAP Summary Plot**: Show the impact of features on model output across all predictions.

### 8. Model Deployment
- **Real-Time Predictions**: Build an API using Flask to deploy the model for real-time machine failure predictions.
- **Dashboard**: Use a dashboard (e.g., Streamlit or Dash) to monitor equipment in real-time, displaying sensor data and failure probabilities.
- **Alert System**: Set thresholds to trigger maintenance alerts when failure probabilities exceed a certain value.

### 9. Report & Documentation
- **Model Performance**: Document the model's performance across different failure modes.
- **Insights**: Highlight key findings, such as the most influential failure mode and the impact of operational conditions (e.g., temperature, torque) on failure rates.
- **Next Steps**: Suggest integration of this system into real-time maintenance processes and possible improvements with additional data.

## Conclusion
This project demonstrates a complete machine learning pipeline using XGBoost to predict failures in industrial equipment. It highlights the importance of feature engineering (sensor data, operational metrics) and the use of interpretable models like XGBoost for industrial predictive maintenance.

## Potential Extensions
- **Real-Time Data Integration**: Incorporate live sensor data streams for continuous monitoring.
- **Cost-Optimization**: Implement cost-sensitive learning to minimize the cost of false positives/negatives based on machine downtime.
- **Anomaly Detection**: Explore unsupervised learning techniques for anomaly detection before failure occurs.


