"""
Classifier: Logistic Regression for IT Support Ticket Priority.
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

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)

sys.path.append(src_dir)

# 1. Load cleaned data
print("Loading data...")
# Assuming you use the same balanced dataset. If you have a specific priority dataset, change the name here.
df = pd.read_csv('../../data/processed/cleaned_tickets.csv')
df = df.dropna(subset=['clean_text', 'Priority'])

X = df['clean_text']
# CHANGED: Target is now the Priority column instead of Department
y = df['Priority']

# 2. Train/test split (Using same random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the Logistic Regression model for Priority
print("Training Logistic Regression model for Priority...")
# class_weight='balanced' is especially important for Priority, as 'High' priority tickets are usually rarer than 'Low'
log_reg_model = LogisticRegression(C=10,random_state=42, max_iter=1000, class_weight='balanced')
log_reg_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = log_reg_model.predict(X_test_tfidf)

print("\n=== Classification Report (Logistic Regression - Priority) ===")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix (visual check)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
# Using Reds for Priority to distinguish it from the Department heatmaps
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Priority')
plt.ylabel('Actual Priority')
plt.title('Confusion Matrix - Logistic Regression (Priority)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../../report_assets/plots/confusion_matrix_priority_logreg.png', dpi=150)
print("Saved confusion_matrix_priority_logreg.png")


# 7. Quick manual test demonstrating probabilities
def predict_priority_with_prob(text, vectorizer, model):
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    classes = model.classes_

    print(f"\nTicket: '{text}'")
    print(f"Predicted Priority: {prediction}")
    print("Confidence breakdown:")

    prob_list = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    for priority, prob in prob_list:
        print(f"  - {priority}: {prob:.2%}")


sample_ticket = "The main database server just crashed and the entire company cannot process any orders!"
predict_priority_with_prob(sample_ticket, vectorizer, log_reg_model)

# 8. Save the model and vectorizer to disk (Updated filenames for Priority)
print("\nSaving Priority models to disk...")
joblib.dump(log_reg_model, '../../prototype/logreg_priority_model.pkl')
joblib.dump(vectorizer, '../../prototype/tfidf_logreg_priority_vectorizer.pkl')
print("Successfully saved logreg_priority_model.pkl and tfidf_logreg_priority_vectorizer.pkl")