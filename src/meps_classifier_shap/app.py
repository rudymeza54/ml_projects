# app.py
import os
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve,
                             average_precision_score)

from shiny import App, reactive, render, ui



# For local development
if os.path.exists(r"C:\Users\rudym\OneDrive\Desktop\training\shiny\hypertension_shap"):
    output_dir = Path(r"C:\Users\rudym\OneDrive\Desktop\training\shiny\hypertension_shap")
else:
    # When deployed to shinyapps.io, use a relative path (current directory)
    output_dir = Path(".")

# Load the saved models and objects
try:
    model = joblib.load(os.path.join(output_dir, "xgboost_model.joblib"))
    explainer = joblib.load(os.path.join(output_dir, "shap_explainer.joblib"))
    MEDICATION_NAMES = joblib.load(os.path.join(output_dir, "medication_names.joblib"))

    # Load datasets from CSV files
    X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(output_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(output_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(output_dir, "y_test.csv"))
    test_df = pd.read_csv(os.path.join(output_dir, "test_df.csv"))
    
    initial_patient = test_df['patient_id'].iloc[0] if not test_df.empty and 'patient_id' in test_df.columns else None

except Exception as e:
    print(f"Error loading files: {e}")
    # Provide fallback behavior or clear error message
    model = None
    explainer = None
    MEDICATION_NAMES = {}
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    test_df = pd.DataFrame()
    initial_patient = None




def get_feature_name(feature):
    """Convert feature name to readable format with default fallback"""
    if feature in MEDICATION_NAMES:
        return MEDICATION_NAMES[feature]
    elif feature.startswith('total_expenditure_'):
        tc_code = feature.replace('total_expenditure_', '')
        tc_class = tc_code.split('_')[0]
        return f"Medication Class {tc_class} (Code {tc_code})"
    return feature


css_path = output_dir

# Make sure the path exists and is valid
print(f"CSS path: {css_path}")
print(f"CSS file exists: {os.path.exists(css_path)}")


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("SHAP Explainer"),
        ui.input_select(
            "plot_type",
            "Plot Type",
            {
                "summary": "Summary Plot",
                "waterfall": "Waterfall Plot",
                "dependence": "Dependence Plot",
            }
        ),
        ui.input_numeric("num_features", "Number of Features", 10, min=5, max=20),
        ui.panel_conditional(
            "input.plot_type === 'waterfall'",
            ui.input_select(
              "patient_id",
              "Patient ID",
              choices=test_df['patient_id'].tolist() if not test_df.empty else ["No patients available"],
              selected=initial_patient if initial_patient in test_df['patient_id'].tolist() else "No patients available"
          )
        ),
        ui.panel_conditional(
            "input.plot_type === 'dependence'",
            ui.input_select(
                "feature",
                "Feature",
                choices=[get_feature_name(col) for col in X_test.columns] if len(X_test.columns) > 0 else []
            )
        ),
        ui.hr(),
        ui.h4("Patient Information"),
        ui.output_ui("patient_stats"),
        ui.hr(),
        ui.h4("Model Parameters"),
        ui.output_ui("model_params"),
        title="SHAP Controls",
        width = "350px",
        class_="bg-light-blue"  # Apply the class from the CSS
    ),
    ui.layout_columns(
        ui.value_box(
            "Accuracy",
            ui.output_text("accuracy"),
            theme="primary",
        ),
        ui.value_box(
            "ROC AUC",
            ui.output_text("roc_auc"),
            theme="info",
        ),
        ui.value_box(
            "Top Feature",
            ui.output_text("top_feature"),
            theme="secondary",
        ),
        ui.value_box(
            "Prediction Method",
            "SHAP",  # Static text instead of radio buttons
            theme="success",
         ),
        width=1/2,
    ),
    # Main visualization area
    ui.card(
        ui.card_header("SHAP Analysis", class_="medical-card-header"),  # Apply the class from the CSS
        ui.output_plot("shap_plot", height="500px", width="100%"),
        full_screen=True,
        class_="medical-card"  # Apply the class from the CSS
    ),
    # # Model metrics card
    # ui.card(
    #     ui.card_header("Model Performance Metrics", class_="medical-card-header"),  # Apply the class from the CSS
    #     ui.output_plot("metrics_plot", height="300px"),
    #     full_screen=True,
    #     class_="medical-card"  # Apply the class from the CSS
    # ),
    # Patient data table
    ui.card(
        ui.card_header("Patient Medication Data", class_="medical-card-header"),  # Apply the class from the CSS
        ui.output_data_frame("patient_data"),
        full_screen=True,
        class_="medical-card"  # Apply the class from the CSS
    ),
    ui.include_css(css_path / "styles.css"),
    title="MEPS Hypertension Prediction SHAP Explainer",
    fillable=True,
)



def server(input, output, session):
    # Reactive values

    @reactive.calc
    def selected_patient_idx():
        # Check if we have data to work with
        if len(X_test) == 0 or test_df.empty:
            print("No patient data available")
            return None
        
        # Get the patient_id from input
        patient_id = input.patient_id()
        
        # Check if a valid patient_id was provided
        if patient_id == "No patients available" or not patient_id:
            print("No valid patient ID selected, using first available patient")
            return 0
        
        try:
            # Make sure test_df has a patient_id column
            if 'patient_id' not in test_df.columns:
                print("'patient_id' column not found in test_df")
                print(f"Available test_df columns: {', '.join(test_df.columns[:5])}...")
                return 0
            
            # Convert patient_id to string for comparison
            patient_id = str(patient_id)
            
            # Find the patient in test_df first
            matching_rows = test_df[test_df['patient_id'].astype(str) == patient_id]
            
            if not matching_rows.empty:
                # Get the position in test_df
                test_df_idx = matching_rows.index[0]
                
                # We need to find the corresponding position in X_test
                # Since the indices don't match, we need to use the common index values
                common_indices = set(X_test.index).intersection(set(test_df.index))
                
                if test_df_idx in common_indices:
                    # The index exists in both dataframes
                    print(f"Found matching patient at shared index: {test_df_idx}")
                    # Get the position (not index) in X_test
                    x_test_position = X_test.index.get_loc(test_df_idx)
                    return x_test_position
                else:
                    # Index doesn't exist in X_test, try to get the position
                    print(f"Patient index {test_df_idx} not in X_test indices, using first patient")
                    return 0
            else:
                # No match found, use first patient as fallback
                print(f"Patient ID '{patient_id}' not found in dataset")
                return 0
                    
        except Exception as e:
            print(f"Error finding patient index: {e}")
            print(f"Debug - X_test shape: {X_test.shape}, test_df shape: {test_df.shape}")
            # Always fall back to the first patient
            return 0
    
    @reactive.calc
    def selected_patient_data():
        idx = selected_patient_idx()
        if idx is None:
            return pd.Series() if len(X_test) == 0 else X_test.iloc[0].copy() * 0
        return X_test.iloc[idx]
    
    @reactive.calc
    def feature_names():
        return [get_feature_name(col) for col in X_test.columns] if len(X_test.columns) > 0 else []
    
    @reactive.calc
    def selected_feature_idx():
        if input.feature() is None:
            return 0
        feature_names_list = feature_names()
        if not feature_names_list:
            return 0
        try:
            return feature_names_list.index(input.feature())
        except ValueError:
            return 0
    
    # Renders
    @render.text
    def accuracy():
        if model is None or len(X_test) == 0 or len(y_test) == 0:
            return "N/A"
        
        # Use model's direct predictions
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test.values.ravel(), y_pred) * 100
        return f"{acc:.1f}%"
    
    @render.ui
    def model_params():
        if not hasattr(model, 'get_params'):
            return ui.p("No model parameters available")
        
        # Get key parameters to display
        params = model.get_params()
        important_params = ['n_estimators', 'max_depth', 'learning_rate', 
                            'min_child_weight', 'subsample', 'colsample_bytree']
        
        # Create a list of UI paragraph elements
        param_elements = []
        for param in important_params:
            if param in params:
                param_elements.append(ui.p(ui.strong(f"{param}: "), f"{params[param]}"))
        
        # Return a div containing all parameter elements
        return ui.div(*param_elements)

    @render.text
    def roc_auc():
        if model is None or len(X_test) == 0 or len(y_test) == 0:
            return "N/A"
        
        # Now using pred_method() instead of input.pred_method()
        # Always use SHAP-based predictions since that's our only option now
        shap_pred = test_df['shap_prediction'].values
        auc = roc_auc_score(y_test, shap_pred)*100
        return f"{auc:.1f}%"
    
    @render.text
    def top_feature():
        try:
            idx = selected_patient_idx()
            if idx is None or idx >= len(X_test):
                return "N/A"
                
            # Generate SHAP values for this specific patient
            single_patient = X_test.iloc[idx:idx+1]
            shap_values = explainer.shap_values(single_patient)[0]  # Get the SHAP values array
            
            # Calculate absolute SHAP values
            abs_shap_values = np.abs(shap_values)
            
            # Get feature names
            names = feature_names()
            
            # Ensure we have both SHAP values and feature names
            if len(abs_shap_values) > 0 and names and len(names) == len(abs_shap_values):
                # Find the index of the top feature
                top_idx = np.argmax(abs_shap_values)
                return names[top_idx]
            
            return "N/A"
        
        except Exception as e:
            print(f"Error in top_feature: {e}")
            return "N/A"
    


    @render.ui
    def patient_stats():
        idx = selected_patient_idx()
        if idx is None:
            return ui.p("No patient selected")
            
        # Ensure index is valid
        if idx >= len(X_test):
            return ui.p(f"Invalid patient index: {idx}")
            
        try:
            patient_data = selected_patient_data()
            
            # Check if 'AGE22X' is in the patient data index
            if 'AGE22X' not in patient_data.index:
                return ui.p("Invalid patient data structure")
                
            age = patient_data['AGE22X']
            
            # Count medications
            med_count = sum(1 for col in patient_data.index 
                           if col.startswith('total_expenditure_') and patient_data[col] > 0)
            
            # Get model prediction
            model_pred_value = model.predict(X_test.iloc[idx:idx+1])[0]
            
            # Get SHAP prediction (log-odds)
            shap_values = explainer.shap_values(X_test.iloc[idx:idx+1])[0]
            shap_log_odds = explainer.expected_value + np.sum(shap_values)
            
            # Convert SHAP log-odds to probability
            shap_prob = 1 / (1 + np.exp(-shap_log_odds))
            
            # Always use SHAP prediction now
            pred_value = shap_prob
            
            # Make sure idx is valid for y_test as well
            actual_value = y_test.iloc[idx].values[0] if idx < len(y_test) else 0
            actual = "Yes" if actual_value > 0.5 else "No"
            
            # Risk tier based on probability
            risk_level = ("Low Risk" if pred_value < 0.3 else 
                          "Medium Risk" if pred_value < 0.7 else 
                          "High Risk")
            
            return ui.div(
                ui.p(ui.strong("Age: "), f"{age:.1f}"),
                ui.p(ui.strong("Medications: "), f"{med_count}"),
                ui.p(ui.strong("Hypertension: "), f"{actual}"),
                ui.p(ui.strong("Prediction: "), f"{risk_level} ({pred_value:.2f})"),
                ui.p(ui.strong("SHAP Pred (prob): "), f"{shap_prob:.2f}"),
                ui.p(ui.strong("SHAP Pred (log-odds): "), f"{shap_log_odds:.2f}"),
                ui.p(ui.strong("Model Pred: "), f"{model_pred_value:.2f}")
            )
        except Exception as e:
            return ui.p(f"Error: {str(e)}")
    
    @render.plot
    def shap_plot():
        try:
            plt.figure(figsize=(12, 8))
            
            if input.plot_type() == "summary":
                if model is None or len(X_test) == 0:
                    plt.text(0.5, 0.5, "No data available", horizontalalignment='center',
                            verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
                else:
                    shap_values = explainer.shap_values(X_test)
                    if len(shap_values) == 0:
                        plt.text(0.5, 0.5, "No SHAP values available", horizontalalignment='center',
                                verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
                    else:
                        shap.summary_plot(
                            shap_values,
                            X_test,
                            feature_names=feature_names(),
                            max_display=input.num_features(),
                            show=False
                        )
                
            elif input.plot_type() == "waterfall":
                idx = selected_patient_idx()
                if idx is None:
                    plt.text(0.5, 0.5, "No patient selected", horizontalalignment='center',
                            verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
                else:
                    try:
                        # Check if index is valid for X_test
                        if idx >= len(X_test):
                            plt.text(0.5, 0.5, f"Invalid patient index: {idx}", horizontalalignment='center',
                                    verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
                            return plt.gcf()
                            
                        # Generate SHAP values for this specific patient
                        single_patient = X_test.iloc[idx:idx+1]
                        shap_values = explainer.shap_values(single_patient)
                        
                        # Create waterfall plot
                        plt.clf()
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values[0],
                                base_values=explainer.expected_value,
                                data=single_patient.iloc[0],
                                feature_names=feature_names()
                            ),
                            max_display=input.num_features(),
                            show=False
                        )
                    except Exception as e:
                        plt.text(0.5, 0.5, f"Error in waterfall plot: {str(e)}", 
                                horizontalalignment='center', verticalalignment='center', 
                                transform=plt.gca().transAxes, fontsize=14)
                
            elif input.plot_type() == "dependence":
                if model is None or len(X_test) == 0:
                    plt.text(0.5, 0.5, "No data available", horizontalalignment='center',
                            verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
                else:
                    shap_values = explainer.shap_values(X_test)
                    if len(shap_values) == 0:
                        plt.text(0.5, 0.5, "No SHAP values available", horizontalalignment='center',
                                verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
                    else:
                        feature_idx = selected_feature_idx()
                        
                        plt.clf()
                        shap.dependence_plot(
                            feature_idx,
                            shap_values,
                            X_test,
                            feature_names=feature_names(),
                            show=False
                        )
                
            plt.tight_layout()
            return plt.gcf()
            
        except Exception as e:
            plt.clf()
            plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
            return plt.gcf()
          
          
    @reactive.calc
    
    def pred_method():
      return "shap"
    
    # @render.plot
    # def metrics_plot():
    #     if model is None or len(X_test) == 0 or len(y_test) == 0:
    #         plt.figure(figsize=(10, 6))
    #         plt.text(0.5, 0.5, "No model metrics available", 
    #                  horizontalalignment='center',
    #                  verticalalignment='center', 
    #                  transform=plt.gca().transAxes, 
    #                  fontsize=14)
    #         return plt.gcf()
    #     
    #     # Get predictions based on selected method
    #     if input.pred_method() == "shap":
    #         y_pred_proba = test_df['shap_prediction'].values
    #     else:
    #         y_pred_proba = model.predict(X_test)
    #     
    #     # Create subplots for ROC and Precision-Recall curves
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #     
    #     # Plot ROC curve
    #     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    #     roc_auc = roc_auc_score(y_test, y_pred_proba)
    #     
    #     ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    #     ax1.plot([0, 1], [0, 1], 'k--')
    #     ax1.set_xlim([0.0, 1.0])
    #     ax1.set_ylim([0.0, 1.05])
    #     ax1.set_xlabel('False Positive Rate')
    #     ax1.set_ylabel('True Positive Rate')
    #     ax1.set_title(f'ROC Curve ({input.pred_method().upper()} Predictions)')
    #     ax1.legend(loc="lower right")
    #     
    #     # Plot Precision-Recall curve
    #     precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    #     ap = average_precision_score(y_test, y_pred_proba)
    #     
    #     ax2.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
    #     ax2.set_xlim([0.0, 1.0])
    #     ax2.set_ylim([0.0, 1.05])
    #     ax2.set_xlabel('Recall')
    #     ax2.set_ylabel('Precision')
    #     ax2.set_title(f'Precision-Recall Curve ({input.pred_method().upper()} Predictions)')
    #     ax2.legend(loc="lower left")
    #     
    #     plt.tight_layout()
    #     return fig
    
    @render.data_frame
    def patient_data():
        try:
            idx = selected_patient_idx()
            if idx is None:
                return render.DataGrid(pd.DataFrame({'Medication': [], 'Expenditure': []}))
                
            if idx >= len(X_test):
                return render.DataGrid(pd.DataFrame({'Error': ['Invalid patient index']}))
                
            patient = X_test.iloc[idx].copy()
            
            # Only show non-zero medications
            nonzero_meds = {
                get_feature_name(col): patient[col]
                for col in patient.index 
                if patient[col] > 0 and col.startswith('total_expenditure_')
            }
            
            # Sort by value (descending)
            nonzero_meds = dict(sorted(nonzero_meds.items(), key=lambda x: x[1], reverse=True))
            
            # Convert to DataFrame with formatted values
            med_df = pd.DataFrame({
                'Medication': nonzero_meds.keys(),
                'Expenditure': ['${:.2f}'.format(val) for val in nonzero_meds.values()]
            })
            
            return render.DataGrid(med_df)
        except Exception as e:
            error_df = pd.DataFrame({'Error': [f'Failed to load patient data: {str(e)}']})
            return render.DataGrid(error_df)

app = App(app_ui, server)
