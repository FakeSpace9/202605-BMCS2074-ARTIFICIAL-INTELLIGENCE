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

model_dir = Path(__file__).resolve().parent    
src_dir = model_dir.parent                     
project_root = src_dir.parent

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import print_classification_report, save_report_metrics, plot_confusion_matrix, save_model_and_vectorizer, load_processed_dataset

# 1. Load cleaned data
print("Loading cleaned dataset...")
df = load_processed_dataset()
# Drop any rows where the text or priority might be missing
df = df.dropna(subset=['clean_text', 'Priority'])

X = df['clean_text']
y = df['Priority']


# 2. Train/Test Split (80% training, 20% testing)
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
print("Extracting features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=350000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Initialize and Train the SVM
print("Training the SVM model for Priority (this might take a moment)...")
svm_model = LinearSVC(C = 5, class_weight='balanced', max_iter=5000, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = svm_model.predict(X_test_tfidf)
print_classification_report(y_test, y_pred, "SVM", "Priority")
save_report_metrics(y_test, y_pred, "SVM", "Priority")

# 6. Confusion matrix
labels = sorted(y.unique())
plot_confusion_matrix(y_test, y_pred, labels, "SVM", "svm", "Priority",
                       cmap='Blues', figsize=(8, 6), xtick_ha='center')

# 7. Quick Manual Test
def predict_priority(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# sample_ticket = "The entire main server is completely offline and no one can work!"
sample_ticket = "Can someone help me configure the SAML integration settings for the guest Wi-Fi? No rush."
# sample_ticket = "Massive broadcast storm on the Omada controller, the entire network profile is locked up and switches are dropping."
print("\nSample prediction:", predict_priority(sample_ticket, vectorizer, svm_model))

# 8. Save the model and vectorizer to disk
save_model_and_vectorizer(svm_model, vectorizer, "svm", "Priority")
