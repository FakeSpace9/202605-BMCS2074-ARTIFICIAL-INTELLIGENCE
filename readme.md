# IT Support Ticket Triage AI

This project is an automated Natural Language Processing (NLP) pipeline designed to triage IT support tickets. It categorizes incoming unstructured text into operational departments and predicts their urgency priority level. This prototype was developed for the BMCS2074 Artificial Intelligence coursework at TAR UMT.

## Project Structure

- **`data/`**: Contains the raw Kaggle dataset and the preprocessed `cleaned_tickets.csv` output.
- **`prototype/`**: Stores all serialized models (`.pkl`) and TF-IDF vectorizers generated during training.
- **`report_assets/`**: Holds auto-generated classification metrics (`.csv`) and confusion matrix plots (`.png`).
- **`src/`**: Contains the core Python scripts, including `app.py` for the web interface, `preprocess.py` for data cleaning, and `utils.py` for shared helper functions.
- **`src/models/`**: Houses the individual training scripts for the Naive Bayes, Logistic Regression, and SVM algorithms.

## Quick Start Guide

**1. Setup Environment**
Install the required dependencies to run the NLP pipeline and Streamlit application:
`pip install pandas scikit-learn nltk streamlit matplotlib seaborn joblib`

**2. Execute Preprocessing**
Clean the raw text data by removing PII, filtering custom stopwords, and folding in ticket tags:
`python src/preprocess.py`

**3. Train the Models**
Run the individual model scripts to extract TF-IDF features and train the classifiers. For example, to train the Naive Bayes models:
`python src/models/Train_Naive_Bayes_dept.py`
`python src/models/Train_Naive_Bayes_Priority.py`

**4. Launch the Web Interface**
Start the interactive Streamlit prototype to test live ticket predictions:
`streamlit run src/app.py`
