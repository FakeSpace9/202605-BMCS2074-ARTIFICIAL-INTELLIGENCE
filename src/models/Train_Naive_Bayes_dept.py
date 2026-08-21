"""
Naive Bayes classifier for three-class support-ticket routing.
Uses the consolidated routing labels created by preprocess.py.
Switched from MultinomialNB + sample_weight='balanced' to ComplementNB
(designed for imbalanced text classification) with tuned alpha and
improved TF-IDF features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# 1. Load the cleaned data.  Department is intentionally a three-class
# first-line-routing target; see preprocess.py for the mapping rationale.
print("loading data...")
df = pd.read_csv(project_root / "data"/"processed"/"cleaned_tickets.csv")
df = df.dropna(subset=['clean_text'])

X = df['clean_text']
y = df['Department']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
# - stop_words='english' removes filler words that add noise
# - sublinear_tf=True dampens the effect of very high-frequency terms
# - min_df=2 drops ultra-rare terms/typos that just add sparsity
vectorizer = TfidfVectorizer(
    max_features=48000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True,
    min_df=2
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train ComplementNB with alpha tuning via GridSearchCV
# ComplementNB is specifically designed to correct the "severe assumptions"
# MultinomialNB makes on imbalanced datasets — no manual sample_weight needed.
param_grid = {
    'alpha': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
    'norm': [False, True],
}
grid = GridSearchCV(
    ComplementNB(),
    param_grid,
    # Accuracy is the stated project target. Macro F1 is still reported below.
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid.fit(X_train_tfidf, y_train)

nb_model = grid.best_estimator_


print(f"ComplementNB best CV accuracy: {grid.best_score_:.4f} ({grid.best_params_})")

print(f"CV accuracy: {grid.best_score_:.4f}")
print("\nClass distribution:")
print(y.value_counts())

# 5. Predict and evaluate
y_pred = nb_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nNaive Bayes ({'ComplementNB'}) Overall Accuracy: {accuracy * 100:.2f}%")

print("\n=== Classification Report (Three-Class Routing) ===")
print(classification_report(y_test, y_pred))
save_report_metrics(y_test, y_pred, "Naive Bayes", "Department", project_root)
# 6. Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Routing Queue')
plt.ylabel('Actual Routing Queue')
plt.title('Confusion Matrix - Naive Bayes (Three-Class Routing)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(project_root/"report_assets"/"plots"/"confusion_matrix_department_nb.png", dpi=150)
print("\nSaved confusion_matrix_department_nb.png")

# 7. Quick manual test
def predict_department(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "My laptop screen is completely black and it won't turn on after the update."
print("\nSample prediction:", predict_department(sample_ticket, vectorizer, nb_model))

# 8. Save models
print("\nSaving models to disk...")
joblib.dump(nb_model, project_root/"prototype"/"model"/"nb_department_model.pkl")
joblib.dump(vectorizer, project_root/"prototype"/"vectorizer"/"tfidf_nb_department_vectorizer.pkl")
print("Successfully saved nb_department_model.pkl and tfidf_nb_department_vectorizer.pkl")
