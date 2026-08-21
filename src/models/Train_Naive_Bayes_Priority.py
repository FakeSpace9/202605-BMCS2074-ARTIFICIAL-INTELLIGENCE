"""
Baseline classifier: Multinomial Naive Bayes for Priority prediction.
Uses the original cleaned_tickets.csv.
MultinomialNB learns the observed priority frequencies and is evaluated using
cross-validated accuracy because overall accuracy is the project target.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
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

# 1. Load the ORIGINAL cleaned data
print("Loading data...")
df = pd.read_csv(project_root / "data"/"processed"/"cleaned_tickets.csv")
df = df.dropna(subset=['clean_text', 'Priority']) 

X = df['clean_text']
y = df['Priority']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF feature extraction
# Relying purely on the pre-cleaned data from cleaned_tickets.csv
# Word n-grams capture phrases such as "system down" and "please resolve".
print("Vectorizing text using word TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=20000,      # Focused vocabulary to prevent noise
    ngram_range=(1, 3),      # Capture phrases up to 3 words long
    sublinear_tf=True,       # Apply sublinear tf scaling to soften the impact of very frequent words
    min_df=2                 # Ignore words that only appear in a single ticket
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train MultinomialNB with alpha tuning via GridSearchCV.
print("Training Multinomial NB model with GridSearch...")
param_grid = {'alpha': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]}
grid = GridSearchCV(
    MultinomialNB(),         
    param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid.fit(X_train_tfidf, y_train)

nb_model = grid.best_estimator_
print(f"Best alpha found: {grid.best_params_['alpha']}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

# 5. Predict and evaluate
y_pred = nb_model.predict(X_test_tfidf)

# Calculate and print the exact accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nNaive Bayes (Multinomial) Overall Accuracy: {accuracy * 100:.2f}%")

print("\n=== Classification Report (Priority) ===")
print(classification_report(y_test, y_pred))
save_report_metrics(y_test, y_pred, "Naive Bayes", "Priority", project_root)
# 6. Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Priority')
plt.ylabel('Actual Priority')
plt.title('Confusion Matrix - Multinomial NB (Priority)')
plt.tight_layout()
plt.savefig(project_root/"report_assets"/"plots"/"confusion_matrix_priority_nb.png", dpi=150)
print("Saved confusion_matrix_priority_nb.png")

# 7. Quick manual test
def predict_priority(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "URGENT: The main database server is down and no one can process payments! Please help immediately."
print("\nSample prediction:", predict_priority(sample_ticket, vectorizer, nb_model))

# 8. Save models
print("\nSaving models to disk...")
joblib.dump(nb_model, project_root/"prototype"/"model"/"nb_priority_model.pkl")
joblib.dump(vectorizer, project_root/"prototype"/"vectorizer"/"tfidf_nb_priority_vectorizer.pkl")
print("Successfully saved nb_priority_model.pkl and tfidf_nb_priority_vectorizer.pkl")
