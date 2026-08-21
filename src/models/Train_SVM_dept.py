"""
SVM classifier for Department routing.
Same pipeline as train_naive_bayes.py, swapping MultinomialNB for LinearSVC.
Uses the cleaned_tickets.csv produced by preprocess.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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

sys.path.append(src_dir)
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

# 1. Load cleaned data
print("Loading cleaned dataset...")
df = pd.read_csv(project_root / "data"/"processed"/"cleaned_tickets.csv")
df = df.dropna(subset=['clean_text'])

X = df['clean_text']
y = df['Department']

# 2. Train/test split (same split logic as the Naive Bayes script,
#    same random_state, so results are directly comparable)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization (identical settings to the Naive Bayes script
#    so any difference in results comes from the model, not the features)
vectorizer = TfidfVectorizer(max_features=250000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train SVM
#    class_weight='balanced' automatically up-weights minority departments
#    (General Inquiry, Human Resources, Sales and Pre-Sales) so the model
#    doesn't just default to predicting the biggest class every time.
svm_model = LinearSVC(C = 5,class_weight='balanced', max_iter=1000, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = svm_model.predict(X_test_tfidf)
print_classification_report(y_test, y_pred, "SVM", "Department")
save_report_metrics(y_test, y_pred, "SVM", "Department", project_root)

# 6. Confusion matrix
labels = sorted(y.unique())
plot_confusion_matrix(y_test, y_pred, labels, "SVM", "svm", "Department", project_root,
                       cmap='Green', figsize=(8, 6), xtick_ha='center')
print("Saved confusion_matrix_department_svm.png")

# 7. Quick manual test with a new made-up ticket
def predict_department(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "My laptop screen is completely black and it won't turn on after the update."
print("\nSample prediction:", predict_department(sample_ticket, vectorizer, svm_model))

# 8. Save the model and vectorizer to disk <-- Added saving logic
print("\nSaving models to disk...")
save_model_and_vectorizer(svm_model, vectorizer, "svm", "Department", project_root)