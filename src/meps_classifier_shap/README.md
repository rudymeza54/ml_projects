# MEPS Hypertension Prediction SHAP Explainer

## Overview

This Shiny application provides an interactive dashboard for exploring machine learning predictions of hypertension risk based on MEPS (Medical Expenditure Panel Survey) data. The application uses SHAP (SHapley Additive exPlanations) values to explain the predictions and help users understand the factors contributing to hypertension risk assessment.

## Features

- **High Performance Metrics**: 94.7% accuracy and 97.6% ROC AUC
- **Interactive Visualizations**: Explore different SHAP visualization types
  - Summary Plot: Overview of feature importance across all patients
  - Waterfall Plot: Detailed explanation for individual patient predictions
  - Dependence Plot: Analyze how specific features affect predictions
- **Patient-Level Insights**: Examine individual patient data and risk factors
- **Medication Analysis**: Review medication expenditures and their impact on predictions

## Dashboard Components

### Main Interface

- **Performance Metrics Cards**: Display model accuracy and ROC AUC scores
- **Feature Importance**: Shows top features influencing predictions (Beta Blockers in the example)
- **SHAP Analysis Visualization**: Interactive visualization of feature importance and impact
- **Patient Medication Data**: Table showing medication expenditures for the selected patient

### Sidebar Controls

- **Plot Type Selection**: Choose between Summary, Waterfall, or Dependence plots
- **Feature Count**: Adjust the number of features displayed (5-20)
- **Patient Selection**: For Waterfall plots, select specific patient records to analyze
- **Feature Selection**: For Dependence plots, select specific features to analyze
- **Patient Information Panel**: Displays demographic and prediction data for the selected patient
- **Model Parameters Panel**: Shows key XGBoost model parameters

## Technical Details

The application uses:
- **XGBoost**: For the underlying prediction model
- **SHAP**: For model explainability
- **Pandas/NumPy**: For data manipulation
- **Matplotlib**: For visualization
- **Scikit-learn**: For model evaluation metrics
- **Shiny for Python**: For the interactive web interface

## Usage

### Setup and Installation

1. Ensure you have Python installed with the required packages:
   ```
   pip install shiny pandas numpy xgboost shap matplotlib scikit-learn joblib
   ```

2. Place the model files in the application directory:
   - `xgboost_model.joblib`: Trained XGBoost model
   - `shap_explainer.joblib`: SHAP explainer object
   - `medication_names.joblib`: Dictionary mapping feature names to readable labels
   - Dataset CSV files: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `test_df.csv`

3. Run the application:
   ```
   shiny run app.py
   ```

### Interpreting Results

- **High Risk (>0.7)**: Patients with high probability of hypertension
- **Medium Risk (0.3-0.7)**: Patients with moderate probability of hypertension
- **Low Risk (<0.3)**: Patients with low probability of hypertension

### SHAP Values

- **Positive SHAP Values (Red)**: Features that increase the probability of hypertension
- **Negative SHAP Values (Blue)**: Features that decrease the probability of hypertension
- **Feature Magnitude**: The size of the SHAP value indicates the strength of the feature's impact

## Implementation Notes

- The application checks for existing model files and displays appropriate error messages if files are missing
- Patient IDs are used to track individual records across datasets
- Medication data is displayed with readable names and formatted expenditure values
- The interface is responsive and supports fullscreen viewing of all visualizations

## Customization

- CSS styling is included for a medical-themed interface
- Sidebar width and card layouts can be adjusted in the UI configuration
- Feature mapping can be extended by modifying the `get_feature_name` function

## Dependencies

- Python 3.8+
- Shiny for Python
- XGBoost
- SHAP
- Pandas/NumPy
- Matplotlib
- Scikit-learn
- Joblib

## License

[Include your license information here]

## Author

[Your Name/Organization]