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

# 1. Load the ORIGINAL cleaned data
print("Loading data...")
df = pd.read_csv('cleaned_tickets.csv')
df = df.dropna(subset=['clean_text', 'Priority']) 

X = df['clean_text']
y = df['Priority']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF feature extraction
# Word n-grams capture phrases such as "system down" and "please resolve".
# Character n-grams capture variants such as urgent/urgently and technical
# terms that may be split or misspelled.  Both produce non-negative features,
# so they remain compatible with MultinomialNB.
print("Vectorizing text using word and character TF-IDF features...")
vectorizer = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 3),
        stop_words='english',
        sublinear_tf=True,
        min_df=2,
    )),
    ('char_tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=30000,
        sublinear_tf=True,
        min_df=3,
    )),
])
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train MultinomialNB with alpha tuning via GridSearchCV.
# The test set remains untouched until this selected model is evaluated once.
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

# 6. Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Priority')
plt.ylabel('Actual Priority')
plt.title('Confusion Matrix - Multinomial NB (Priority)')
plt.tight_layout()
plt.savefig('confusion_matrix_priority_nb.png', dpi=150)
print("Saved confusion_matrix_priority_nb.png")

# 7. Quick manual test
def predict_priority(text, vectorizer, model):
    from preprocess import clean_text 
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "URGENT: The main database server is down and no one can process payments! Please help immediately."
print("\nSample prediction:", predict_priority(sample_ticket, vectorizer, nb_model))

# 8. Save models
print("\nSaving models to disk...")
joblib.dump(nb_model, 'nb_priority_model.pkl')
joblib.dump(vectorizer, 'tfidf_priority_vectorizer.pkl')
print("Successfully saved nb_priority_model.pkl and tfidf_priority_vectorizer.pkl")
