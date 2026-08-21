"""
Classifier: Logistic Regression for IT Support Ticket Priority.
Uses the cleaned_tickets_balanced.csv produced by preprocess.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

from pathlib import Path

model_dir = Path(__file__).resolve().parent    # .../src/models
src_dir = model_dir.parent                     # .../src
project_root = src_dir.parent
from utils import print_classification_report, save_report_metrics, plot_confusion_matrix, save_model_and_vectorizer


def save_report_metrics(y_true, y_pred, model_name, task_name, root_path):
    """
    Extracts the classification report and saves it as a CSV 
    for easy copy-pasting into the assignment report tables.
    """
    # Generate the report as a dictionary
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert to a Pandas DataFrame and round to 4 decimal places
    df_metrics = pd.DataFrame(report_dict).transpose().round(4)
    
    # Ensure the output directory exists
    output_dir = root_path / "report_assets" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    filename = f"{model_name.replace(' ', '_')}_{task_name}_metrics.csv"
    file_path = output_dir / filename
    df_metrics.to_csv(file_path)
    
    print(f"✅ Saved {model_name} metrics for your report to: {file_path}")

sys.path.append(src_dir)

# 1. Load cleaned data
print("Loading data...")
# Assuming you use the same balanced dataset. If you have a specific priority dataset, change the name here.
df = pd.read_csv(project_root / "data"/"processed"/"cleaned_tickets.csv")
df = df.dropna(subset=['clean_text', 'Priority'])

X = df['clean_text']
# CHANGED: Target is now the Priority column instead of Department
y = df['Priority']

# 2. Train/test split (Using same random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the Logistic Regression model for Priority
print("Training Logistic Regression model for Priority...")
# class_weight='balanced' is especially important for Priority, as 'High' priority tickets are usually rarer than 'Low'
log_reg_model = LogisticRegression(C=10,random_state=42, max_iter=1000, class_weight='balanced')
log_reg_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = log_reg_model.predict(X_test_tfidf)
print_classification_report(y_test, y_pred, "Logistic Regression", "Priority")
save_report_metrics(y_test, y_pred, "Logistic Regression", "Priority", project_root)
# 6. Confusion matrix (visual check)
labels = sorted(y.unique())
plot_confusion_matrix(y_test, y_pred, labels, "Logistic Regression", "logreg", "Priority", project_root, cmap='Reds', figsize=(10, 8))



# 7. Quick manual test demonstrating probabilities
def predict_priority_with_prob(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    classes = model.classes_

    print(f"\nTicket: '{text}'")
    print(f"Predicted Priority: {prediction}")
    print("Confidence breakdown:")

    prob_list = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    for priority, prob in prob_list:
        print(f"  - {priority}: {prob:.2%}")


sample_ticket = "The main database server just crashed and the entire company cannot process any orders!"
predict_priority_with_prob(sample_ticket, vectorizer, log_reg_model)

# 8. Save the model and vectorizer to disk (Updated filenames for Priority)
save_model_and_vectorizer(log_reg_model, vectorizer, "logreg", "Priority", project_root)