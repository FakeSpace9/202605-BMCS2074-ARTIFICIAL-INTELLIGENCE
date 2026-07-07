"""
Baseline classifier: Multinomial Naive Bayes for Department routing.
Uses the cleaned_tickets.csv produced by preprocess.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # <-- Added this import

# 1. Load cleaned data
df = pd.read_csv('cleaned_tickets_balanced.csv')
df = df.dropna(subset=['clean_text'])  # safety net in case any cleaned to empty string

X = df['clean_text']
y = df['Department']

# 2. Train/test split (stratify so rare departments appear in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
#    max_features caps vocabulary size; ngram_range=(1,2) also captures
#    two-word phrases like "account management" not just single words
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # transform only, never fit on test data

# 4. Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = nb_model.predict(X_test_tfidf)

print("=== Classification Report (Department) ===")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix (visual check of which departments get confused)
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

# 7. Quick manual test with a new made-up ticket
def predict_department(text, vectorizer, model):
    from preprocess import clean_text  # reuse your cleaning function
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "My laptop screen is completely black and it won't turn on after the update."
print("\nSample prediction:", predict_department(sample_ticket, vectorizer, nb_model))

# 8. Save the model and vectorizer to disk <-- Added saving logic
print("\nSaving models to disk...")
joblib.dump(nb_model, 'nb_department_model.pkl')
joblib.dump(vectorizer, 'tfidf_department_vectorizer.pkl')
print("Successfully saved nb_department_model.pkl and tfidf_department_vectorizer.pkl")