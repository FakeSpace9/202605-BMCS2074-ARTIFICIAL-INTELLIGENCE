"""
Baseline classifier: Naive Bayes for Department routing.
Uses the original cleaned_tickets.csv.
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

# 1. Load the ORIGINAL cleaned data
df = pd.read_csv('cleaned_tickets.csv')
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
    max_features=15000,
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
param_grid = {'alpha': [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]}
grid = GridSearchCV(
    ComplementNB(),
    param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)
grid.fit(X_train_tfidf, y_train)

nb_model = grid.best_estimator_
print(f"Best alpha found: {grid.best_params_['alpha']}")
print(f"Best CV f1_macro: {grid.best_score_:.4f}")

# 5. Predict and evaluate
y_pred = nb_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nNaive Bayes (Complement) Overall Accuracy: {accuracy * 100:.2f}%")

print("\n=== Classification Report (Department) ===")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Department')
plt.ylabel('Actual Department')
plt.title('Confusion Matrix - Naive Bayes (Department)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_department_nb.png', dpi=150)
print("\nSaved confusion_matrix_department_nb.png")

# 7. Quick manual test
def predict_department(text, vectorizer, model):
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "My laptop screen is completely black and it won't turn on after the update."
print("\nSample prediction:", predict_department(sample_ticket, vectorizer, nb_model))

# 8. Save models
print("\nSaving models to disk...")
joblib.dump(nb_model, 'nb_department_model.pkl')
joblib.dump(vectorizer, 'tfidf_department_vectorizer.pkl')
print("Successfully saved nb_department_model.pkl and tfidf_department_vectorizer.pkl")