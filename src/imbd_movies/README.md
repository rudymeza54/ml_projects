# IMDB Movies Challenge

## Overview
This project focuses on performing ETL (Extract, Transform, Load) operations and conducting in-depth data analysis on a large IMDB movies dataset. A key aspect of the analysis is a comparative study of Quentin Tarantino’s films against the broader dataset.

## Project Structure

### Part 1: Data Setup and ETL
- Deployment of Docker containers for streamlined data processing and storage
- Loading the IMDB movies dataset into PostgreSQL
- Development of an optimized data pipeline to enhance Tableau performance
- Utilization of PySpark for ETL operations

### Part 2: Data Analysis and Visualization
- Conducting Exploratory Data Analysis (EDA) 
- Implementing feature selection techniques
- Comparative analysis of all directors versus Quentin Tarantino
- Application of machine learning using LASSO regression modeling
- Preparation of subset data for Tableau dashboard visualizations

## Technical Architecture
The project leverages a containerized infrastructure using Docker:
- **Jupyter/PySpark notebook container** for data processing
- **PostgreSQL container** for efficient database storage
- **Docker network (“MovieNetwork”)** facilitating seamless container communication
- **VS Code integration** for enhanced development workflow

## Data Processing Workflow

1. **Environment Setup**:
   - Establish Docker network
   - Deploy Jupyter/PySpark and PostgreSQL containers
   - Transfer IMDB dataset (5,129,693 rows) and PostgreSQL driver to containers

2. **ETL Operations**:
   - Initiate Spark session and connect to PostgreSQL
   - Load and transform IMDB movies dataset
   - Construct SQL queries for data analysis
   - Store processed data in PostgreSQL

3. **Data Analysis**:
   - Comparative rating distribution analysis (All Directors vs. Tarantino)
   - Runtime analysis (mean, minimum, maximum comparisons)
   - Genre-based evaluations utilizing log-scale visualizations
   - Examination of production company influence
   - Development of time-series visualizations for releases and ratings

4. **Machine Learning Implementation**:
   - Feature extraction and preprocessing using PySpark ML
   - LASSO regularized logistic regression modeling
   - Cross-validation and model evaluation (AUC, Accuracy, Precision, Recall)
   - Visualization of feature importance to determine key drivers of high-rated Tarantino films

## Key Visualizations

- IMDB rating distributions
- Runtime comparative analysis
- Genre preference distribution
- Production company influence
- Time-series trends of movie releases and ratings
- Feature importance evaluation in machine learning model

## Tech Stack
- **Docker** – Containerization and environment management
- **PySpark** – Large-scale data processing and SQL-based operations
- **PostgreSQL** – Efficient data storage and querying
- **Pandas** – Data manipulation and preprocessing
- **Matplotlib & Seaborn** – Data visualization and trend analysis
- **PySpark ML** – Machine learning model implementation
- **Tableau** – (Planned) Interactive dashboard visualization

## Author
**Rudy Meza**  
- **Part 1 Published:** October 27, 2024  
- **Part 2 Published:** October 30, 2024  

## Future Directions
The final dataset will be leveraged to construct a Tableau dashboard that highlights significant insights derived from the analysis. The primary focus will be on evaluating Quentin Tarantino’s films within the broader industry context.

