# Data Science Portfolio - Rudy Meza

## Project Overview

This repository contains a collection of advanced data science projects focusing on healthcare analytics and entertainment industry analysis. Each project demonstrates expertise in different aspects of the data science lifecycle - from ETL pipelines and database management to machine learning model development and interactive visualization dashboards.

## Key Projects

### MEPS Hypertension Prediction SHAP Explainer

An interactive Shiny application that visualizes machine learning predictions for hypertension risk using Medical Expenditure Panel Survey (MEPS) data.

**Technical Highlights:**
- XGBoost predictive model achieving 94.7% accuracy and 97.6% ROC AUC
- SHAP (SHapley Additive exPlanations) implementation for model interpretability
- Interactive dashboard with three visualization types (Summary, Waterfall, and Dependence plots)
- Patient-level medication analysis with expenditure breakdowns
- Python-based Shiny web application for healthcare professionals

**Key Features:**
- Real-time exploration of feature importance with Beta Blockers as top predictor
- Individual patient risk analysis with detailed explanations
- Medication impact visualization showing how different prescriptions affect risk
- Customizable visualizations with adjustable parameters

## Live Demo

Explore the live demo of the Tariff Impact Analysis Dashboard:  
👉 [Shiny App]([https://tariffimpact.netlify.app/](https://rudy-meza.shinyapps.io/hypertension_shap1/))


### IMDB Movies Challenge

A comprehensive ETL and analysis project examining a large-scale IMDB dataset (5.1M+ rows) with a special focus on comparing Quentin Tarantino's filmography against industry standards.

**Technical Highlights:**
- Docker-containerized architecture with Jupyter/PySpark and PostgreSQL
- Advanced ETL pipeline for processing multi-million row datasets
- Feature selection and LASSO regularized logistic regression modeling
- Cross-validation and model evaluation (AUC, Accuracy, Precision, Recall)
- Time-series analysis of film releases and ratings

**Key Insights:**
- Statistical comparison of Tarantino films versus industry averages
- Runtime distribution analysis
- Genre preference patterns on log-scale
- Production company involvement metrics
- Feature importance visualization identifying key factors in high-rated Tarantino films

## Technical Skills Demonstrated

- **Big Data Processing**: PySpark, SQL
- **Databases**: PostgreSQL
- **Containerization**: Docker
- **Machine Learning**: XGBoost, LASSO regression, PySpark ML
- **Explainable AI**: SHAP library
- **Data Visualization**: Matplotlib, Seaborn, Shiny
- **Web Applications**: Shiny for Python
- **Statistical Analysis**: Distribution comparisons, time-series analysis

## Installation and Usage

Each project contains its own setup instructions in its respective directory:

- `meps-hypertension/`: MEPS Hypertension Prediction SHAP Explainer
- `imdb-movies/`: IMDB Movies Challenge

## Author

Rudy Meza

## License

[Specify your license information]
