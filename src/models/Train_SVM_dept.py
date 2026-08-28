import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
from pathlib import Path

model_dir = Path(__file__).resolve().parent    
src_dir = model_dir.parent                     
project_root = src_dir.parent
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import print_classification_report, save_report_metrics, plot_confusion_matrix, save_model_and_vectorizer, load_processed_dataset




# 1. Load cleaned data
print("Loading cleaned dataset...")
df = load_processed_dataset()
df = df.dropna(subset=['clean_text'])

X = df['clean_text']
y = df['Department']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization (identical settings to the Naive Bayes script
vectorizer = TfidfVectorizer(max_features=200000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train SVM
svm_model = LinearSVC(C = 5,class_weight='balanced', max_iter=5000, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = svm_model.predict(X_test_tfidf)
print_classification_report(y_test, y_pred, "SVM", "Department")
save_report_metrics(y_test, y_pred, "SVM", "Department")

# 6. Confusion matrix
labels = sorted(y.unique())
plot_confusion_matrix(y_test, y_pred, labels, "SVM", "svm", "Department",
                       cmap='Greens', figsize=(8, 6), xtick_ha='center')
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

# 8. Save the model and vectorizer to disk
print("\nSaving models to disk...")
save_model_and_vectorizer(svm_model, vectorizer, "svm", "Department")