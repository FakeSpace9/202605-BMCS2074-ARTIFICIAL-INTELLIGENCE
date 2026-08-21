"""
Upgraded Classifier: Logistic Regression for Department routing.
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

# 1. Load cleaned data
print("Loading data...")
df = pd.read_csv(project_root / "data"/"processed"/"cleaned_tickets.csv")
df = df.dropna(subset=['clean_text'])

X = df['clean_text']
y = df['Department']

# 2. Train/test split (Exact same random_state to ensure fair comparison with baseline and SVM)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization (The Feature Extraction Step)
print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=27000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the Logistic Regression model
print("Training Logistic Regression model...")
# max_iter is set to 1000 to ensure the solver converges on high-dimensional text data
log_reg_model = LogisticRegression(C=10,random_state=42, max_iter=1000, class_weight='balanced')
log_reg_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = log_reg_model.predict(X_test_tfidf)

print("\n=== Classification Report (Logistic Regression - Department) ===")
print(classification_report(y_test, y_pred))
save_report_metrics(y_test, y_pred, "Logistic Regression", "Department", project_root)
# 6. Confusion matrix (visual check)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
# Using Greens for the heatmap so it looks distinct from the Naive Bayes (Blues) and SVM
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Department')
plt.ylabel('Actual Department')
plt.title('Confusion Matrix - Logistic Regression (Department)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(project_root/"report_assets"/"plots"/"confusion_matrix_department_logreg.png", dpi=150)
print("Saved confusion_matrix_department_logreg.png")


# 7. Quick manual test demonstrating probabilities
def predict_department_with_prob(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    # Get the predicted class
    prediction = model.predict(vec)[0]

    # Get the probabilities for all classes
    probabilities = model.predict_proba(vec)[0]
    classes = model.classes_

    print(f"\nTicket: '{text}'")
    print(f"Predicted Department: {prediction}")
    print("Confidence breakdown:")

    # Pair classes with their probabilities and sort them highest to lowest
    prob_list = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    for dept, prob in prob_list:
        if prob > 0.01:  # Only show departments with > 1% probability
            print(f"  - {dept}: {prob:.2%}")


sample_ticket = "My laptop screen is completely black and it won't turn on after the update."
predict_department_with_prob(sample_ticket, vectorizer, log_reg_model)

# 8. Save the model and vectorizer to disk
print("\nSaving models to disk...")
joblib.dump(log_reg_model, project_root/"prototype"/"model"/"logreg_department_model.pkl")
joblib.dump(vectorizer, project_root/"prototype"/"vectorizer"/"tfidf_logreg_department_vectorizer.pkl")
print("Successfully saved logreg_department_model.pkl and tfidf_logreg_department_vectorizer.pkl")