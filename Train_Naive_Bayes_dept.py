"""
Baseline classifier: Multinomial Naive Bayes for Department routing.
Uses the original cleaned_tickets.csv and algorithmic sample weights.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight  # <-- Added import
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
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Naive Bayes with Sample Weights to fix imbalance internally
nb_model = MultinomialNB()
# Calculate weights dynamically based on the training labels
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
# Fit the model using these weights
nb_model.fit(X_train_tfidf, y_train, sample_weight=sample_weights)

# 5. Predict and evaluate
y_pred = nb_model.predict(X_test_tfidf)

print("=== Classification Report (Department) ===")
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
print("Saved confusion_matrix_department_nb.png")

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