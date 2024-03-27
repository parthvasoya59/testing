
# Real-Time Detection of Financial Fraud: Big Data Analytics in Action

This project aims to develop a robust system for detecting fraudulent credit card transactions in real-time using machine learning techniques. The project follows a multi-step approach:

Model Training and Selection: Initially, the dataset was trained and tested on 10 different machine learning algorithms to identify the most suitable model. This process was documented in the model_creation.ipynb notebook, which was executed on Google Colab. Metrics such as prediction time, accuracy, F1-score, and recall were used to evaluate the performance of each algorithm. Based on these metrics, the Random Forest algorithm was selected as the final model.

Hyperparameter Tuning and Final Model Creation: Following the selection of the Random Forest algorithm, hyperparameter tuning was performed to optimize its performance further. The tuned model was then saved using the Joblib library. This process was documented in the model_creation_final.ipynb notebook, also executed on Google Colab.

Real-time Prediction Pipeline: A real-time prediction pipeline was implemented using Apache Beam. This pipeline processes incoming transaction data from various sources, including Amazon S3 and local files. The data is then analyzed to determine whether each transaction is valid or fraudulent. The results are stored and displayed on a PowerBI dashboard for monitoring purposes.

Streamlit Dashboard: In addition to the PowerBI dashboard, a Streamlit web application was developed to provide a user-friendly interface for uploading transaction files. Users can upload files containing transactions, which are then used for prediction.


## Table of Contents

* [Installation](#installation)
* [Running the System](#Running-the-System) 
* [Usage](#usage)
* [Model Training](#model-training)
* [Real-time Prediction Pipeline](#real-time-prediction-pipeline)
* [Results](#results)       
## Installation

This project requires a specific Python version and dependencies. To set up the environment, follow these steps:

1. **Create a conda environment:**

   ```bash
   conda create -p venv python==3.10 -y
   ```

   This command creates a new conda environment named `venv` (you can choose a different name if you prefer) and installs Python version 3.10 within that environment. The `-y` flag tells conda to proceed without prompting for confirmation.

2. **Activate the conda environment:**

   The activation command depends on your operating system:

   - **Windows:**

     ```bash
     conda activate K:\ResearchProject\project\code\venv
     ```

   - **macOS/Linux:**

     ```bash
     source K:/ResearchProject/project/code/venv/bin/activate
     ```

   Replace `K:\ResearchProject\project\code\venv` with the actual path to your conda environment. This activates the environment you just created, making its Python and packages available for use.

3. **Install project dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This command installs the dependencies listed in your project's `requirements.txt` file using pip, the package installer for Python. Make sure you have this file in your project directory.

**Important:**

- Make sure you have conda installed before running these commands. You can download it from the Anaconda website (https://www.anaconda.com/).
- Replace `K:\ResearchProject\project\code\venv` with the actual path to your conda environment on your system.

By following these steps, you'll have a clean environment with all the necessary dependencies installed to run your project.
## Running the System

This section explains how to launch the credit card fraud detection system:

**1. Streamlit App (Interactive File Upload and Prediction):**

To utilize the interactive Streamlit app for uploading transaction files and viewing predictions, use the following command in your terminal:

```bash
streamlit run .\streamlitapp.py
```

This command starts the Streamlit app defined in `streamlitapp.py`. This app allows you to upload individual transaction files and provides visual feedback on the predicted outcomes (fraudulent or legitimate).

**2. Real-time Processing Script (app.py):**

If you desire continuous, real-time processing of transaction data, execute the script responsible for this functionality:

```bash
python .\app.py
```

This command runs the Python script `app.py`. This script is likely designed for real-time operation, potentially utilizing Apache Beam for data ingestion, processing, and prediction. It wouldn't provide an interactive interface but would continuously monitor and process incoming transaction data.

**Key Points:**

* Choose the Streamlit app for interaction with individual transactions.
* Use `app.py` for real-time, continuous processing of transaction streams.
## Usage

This section outlines the steps to use this project for real-time transaction prediction:

**1. Model Training and Selection:**

* **Train and Test Models:**
    * Open the `model_creation.ipynb` notebook in Google Colab.
    * This notebook trains and tests the dataset on 10 different machine learning algorithms.
* **Analyze Performance:**
    * Analyze the generated performance metrics (accuracy, prediction time, F1-score, recall) to identify the best performing algorithm.
    * Based on the analysis, the notebook is expected to conclude that the Random Forest algorithm is the most suitable model for this task.

**2. Final Model Generation and Hyperparameter Tuning:**

* **Hyperparameter Tuning:**
    * Open the `model_creation_final.ipynb` notebook in Google Colab.
    * This notebook performs hyperparameter tuning on the chosen Random Forest model to optimize its performance.
* **Generate Final Model File:**
    * The notebook utilizes Joblib to save the final, optimized Random Forest model as a file.

**3. Real-time Prediction Pipeline:**

* **Data Ingestion:**
    * An Apache Beam pipeline is set up to process incoming transaction data from various sources like:
        * Amazon S3 buckets
        * Local filesystems on servers

**4. Choose Your Prediction Interface:**

Here, you have two options for interacting with the prediction system:

* **Streamlit App (For Single Transactions):**
    * If you have individual transaction records and want immediate predictions, use the Streamlit app.
    * Deploy the Streamlit app to provide a user interface where users can upload transaction files for prediction.

* **Power BI Dashboard (For Continuous Data Processing from S3):**
    * If you're dealing with continuous data streams from Amazon S3, use the Power BI dashboard for visualization.
    * The Apache Beam pipeline feeds the incoming data directly to the Power BI dashboard for processing and real-time prediction updates.

**5. Monitoring and Analysis (Optional):**

* **Prediction Monitoring (For Streamlit App):**
    * Regardless of which interface you choose, the system allows for monitoring the generated predictions.
* **Power BI Dashboard Visualization (For Continuous Data):**
    * When using Power BI, a dashboard is automatically displayed, allowing you to visualize and analyze the prediction results in real-time.

**Note:** This explanation assumes the existence of two Jupyter notebooks (`model_creation.ipynb` and `model_creation_final.ipynb`) within the project.
## Model Training

This section outlines the process of training the model for credit card transaction fraud detection:

**1. Data Acquisition:**

* Obtain the credit card fraud detection dataset from Kaggle. You can find several relevant datasets by searching on Kaggle ([https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

**2. Data Preprocessing:**

* **Data Cleaning:** Handle missing values using appropriate techniques like imputation or deletion.
* **Class Imbalance:** Address the potential issue of class imbalance (where fraudulent transactions are a small fraction of the total data). Techniques like SMOTE (Synthetic Minority Oversampling Technique) can be used to balance the classes.
* **Feature Scaling:** Scale features like 'Time' and 'Amount' for better training performance. This ensures all features contribute proportionally during model training.

**3. Exploratory Data Analysis (EDA):**

* **Visualize Class Distribution:** Analyze the distribution of fraudulent and legitimate transactions to understand the dataset's skewness. Techniques like histograms or bar charts can be helpful here.
* **Feature Distribution and Correlations:** Analyze the distribution of individual features and identify any correlations between them. This helps understand the relationships between features and potential redundancies.

**4. Feature Engineering:**

* **'Time' Feature Transformation:** Consider transforming the 'Time' feature to more meaningful units like minutes or hours based on your analysis.
* **Transaction Patterns:** Investigate transaction patterns over different time intervals, such as weekdays/weekends or specific times of day. This could potentially reveal insights related to fraudulent activity.

**5. Dimensionality Reduction (Optional):**

* **PCA:** Apply Principal Component Analysis (PCA) if the dataset has a high number of features. PCA helps reduce dimensionality while preserving the most important information.

**6. Model Training and Evaluation:**

* Train multiple machine learning models on the preprocessed data. You mentioned using  'Random Forest', 'Naive Bayes', 'Artificial Neural Network', 'Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbors', 'AdaBoost', 'Gradient Boosting', 'Decision Tree', 'Extra Trees', 'XGBoost', and 'LightGBM'.
* Evaluate model performance based on metrics like accuracy, precision, recall, and F1-score. Choose the model that performs best based on these metrics.

**Key Takeaway:**

* We identified the **Random Forest** algorithm as the best performing model based on the evaluation process.
 
**Real-time Prediction Pipeline**

This section details the real-time prediction pipeline for credit card transaction fraud detection, offering two user-friendly options for data ingestion and visualization:

**Data Sources:**

The system can ingest transaction data from various sources, providing flexibility to your workflow:

* **Amazon S3 Buckets:** Leverage cloud storage for continuous data streams.
* **Local Filesystems:** Utilize on-premise data stored on servers.

**Apache Beam Pipeline:**

A core component of the pipeline, Apache Beam, efficiently processes and transforms the incoming transaction data in real-time. This ensures smooth data handling before prediction.

**Prediction:**

The trained Random Forest model is applied to the preprocessed transactions to predict their legitimacy (fraudulent or legitimate). This prediction capability enables real-time detection of potential fraudulent activities.

**Choose Your Visualization and User Interaction:**

The system empowers you to select the most suitable approach for visualization and user interaction based on your needs:

* **Streamlit App (For Single Transactions):**
    * This user-friendly web application provides a convenient way for users to upload individual transaction files.
    * The Streamlit app interface displays the predicted outcome (fraudulent or legitimate) for each uploaded transaction.
    * Ideal for scenarios where users need to analyze individual transactions.

* **Power BI Dashboard (For Continuous Data Processing from S3):**
    * When dealing with continuous data streams from Amazon S3, the Power BI dashboard provides real-time visualization of predictions.
    * The Apache Beam pipeline seamlessly feeds the incoming data directly into the Power BI dashboard. This allows for automatic updates to the visualizations, reflecting the latest predictions.
    * Best suited for continuous monitoring and analysis of large datasets.

**Benefits of Combined Approach:**

This flexible design offers several advantages:

* **Scalability:** The system can handle various data sources and volumes, adapting to your specific requirements.
* **User Choice:** You can select the visualization and interaction method that aligns best with your workflow (individual file analysis or continuous data monitoring).
* **Real-time Detection:** The pipeline enables real-time prediction, empowering you to identify potential fraudulent transactions as they occur.

By combining these elements, you gain a robust and user-friendly system for credit card fraud detection, enhancing the security of your credit card transactions.
## Results

This section presents the outcomes of the credit card fraud detection system:

**Model Performance:**

The system evaluated various machine learning algorithms on the credit card fraud dataset. Here's a detailed breakdown of the performance metrics for each model:

| Model | Accuracy | Precision | Recall | F1-score | Prediction Time (s) | AUC |
|---|---|---|---|---|---|---|
| Random Forest (**Final Model**) | 0.9996 | 0.9011 | 0.8367 | 0.8677 | 2.2474 | 0.99 |
| Naive Bayes | 0.8756 | 0.9540 | 0.7899 | 0.8642 | 0.0738 | 0.94 |
| Artificial Neural Network | 0.7727 | 0.9529 | 0.5747 | 0.7169 | 0.1967 | 0.79 |
| Logistic Regression | 0.9387 | 0.9577 | 0.9181 | 0.9375 | 0.0177 | 0.98 |
| Support Vector Machine | 0.9235 | 0.9581 | 0.8860 | 0.9206 | 370.8935 | 0.96 |
| K-Nearest Neighbors | 0.9076 | 0.9837 | 0.8293 | 0.8999 | 61.1380 | 0.91 |
| AdaBoost | 0.9402 | 0.9609 | 0.9181 | 0.9390 | 0.6614 | 0.99 |
| Gradient Boosting | 0.9523 | 0.9663 | 0.9374 | 0.9517 | 0.2135 | 0.99 |
| Decision Tree | 0.9127 | 0.9758 | 0.8467 | 0.9067 | 0.0160 | 0.91 |
| Extra Trees | 0.9149 | 0.9994 | 0.8306 | 0.9072 | 1.9585 | 0.99 |
| XGBoost | 0.9360 | 0.9936 | 0.8779 | 0.9322 | 0.2187 | 0.99 |
| LightGBM | 0.9423 | 0.9836 | 0.8999 | 0.9399 | 0.7626 | 0.99 |

**Key Observations:**

* **Random Forest:** As the chosen final model, it achieved a very high accuracy (0.9996) and AUC score (0.99), indicating excellent performance in classifying fraudulent and legitimate transactions. However, it has a slightly longer prediction time compared to some other models.
* **Ensemble Methods (AdaBoost, Gradient Boosting, Extra Trees, XGBoost, LightGBM):** These algorithms delivered impressive performance with AUC scores of 0.99, demonstrating strong classification abilities. They might be worth exploring further depending on your specific requirements for accuracy and prediction speed.
* **Logistic Regression and Support Vector Machine:** These traditional models exhibited strong results with AUC scores of 0.98 and 0.96, respectively. They offer a good balance between accuracy and prediction time.
* **Decision Tree:** This model provided a respectable AUC score (0.91) with relatively low prediction time, making it suitable for real-time applications where speed is a priority.
* **Naive Bayes:** While boasting a very low prediction time (ideal for real-time scenarios), its classification performance may not be as high as other algorithms.

