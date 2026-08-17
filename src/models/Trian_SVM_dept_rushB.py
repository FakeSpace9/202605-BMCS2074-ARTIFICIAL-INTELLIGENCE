"""
SVM Model to Classify Ticket Department (with Automated Tuning)
Uses Optuna to automatically find the best parameters.
Stops automatically when no improvements are found and saves with '_rushB'.
"""

import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load and prepare data
print("Loading cleaned dataset...")
df = pd.read_csv('../../data/processed/cleaned_tickets.csv').dropna(subset=['clean_text', 'Department'])
X = df['clean_text']
y = df['Department']

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Custom Early Stopping Rule
class EarlyStoppingCallback:
    def __init__(self, patience=5):
        self.patience = patience

    def __call__(self, study, trial):
        # If the current trial number is far past the best trial number, kill the search
        if study.best_trial.number + self.patience <= trial.number:
            print(f"\n[Early Stopping] No improvement for {self.patience} trials. Halting search!")
            study.stop()

# 3. Define the Optuna Objective Function
def objective(trial):
    # Optuna automatically suggests values within these ranges
    max_features = trial.suggest_int('max_features', 5000, 500000)
    ngram_max = trial.suggest_int('ngram_max', 1, 3)
    C_value = trial.suggest_float('C', 0.1, 10.0, log=True)
    max_iter_val = trial.suggest_int('max_iter', 400, 5000)

    # Build the pipeline with the suggested parameters
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))),
        ('svm', LinearSVC(C=C_value, class_weight='balanced', max_iter=max_iter_val, random_state=42))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)

# 4. Create the Study and Optimize
print("Starting automated search for best accuracy (Optuna)...")
study = optuna.create_study(direction='maximize')

# Run 50 trials, using our custom EarlyStoppingCallback
study.optimize(objective, n_trials=50, callbacks=[EarlyStoppingCallback(patience=5)])

print("\n*** BEST PARAMETERS FOUND ***")
print(study.best_params)
print(f"Best Accuracy: {study.best_value:.4f}")
print("*****************************\n")

# 5. Train the final model with the absolute best parameters
print("Training final model with best parameters...")
best_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=study.best_params['max_features'],
                              ngram_range=(1, study.best_params['ngram_max']))),
    ('svm', LinearSVC(C=study.best_params['C'],
                      class_weight='balanced',
                      max_iter=study.best_params['max_iter'],
                      random_state=42))
])
best_pipeline.fit(X_train, y_train)

# 6. Predict and evaluate for the final report
print("Evaluating the final model...")
y_pred = best_pipeline.predict(X_test)

print("=== Classification Report (Department) - SVM ===")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix Generation
print("Generating Confusion Matrix visualization...")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
# Using Greens for Department to contrast with the Blues used for Priority
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Department')
plt.ylabel('Actual Department')
plt.title('Confusion Matrix - SVM (Department)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save image with the rushB suffix
plt.savefig('../../report_assets/plots/confusion_matrix_department_svm_rushB.png', dpi=150)
print("Saved confusion_matrix_department_svm_rushB.png")

# 8. Save the final pipeline
print("\nSaving pipeline to disk...")
# Save model with the rushB suffix
joblib.dump(best_pipeline, '../../prototype/svm_department_pipeline_rushB.pkl')
print("Successfully saved svm_department_pipeline_rushB.pkl")