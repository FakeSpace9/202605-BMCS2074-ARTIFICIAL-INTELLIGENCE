"""
SVM classifier for Department routing.
Same pipeline as train_naive_bayes.py, swapping MultinomialNB for LinearSVC.
Uses the cleaned_tickets.csv produced by preprocess.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load cleaned data
df = pd.read_csv('cleaned_tickets.csv')
df = df.dropna(subset=['clean_text'])

X = df['clean_text']
y = df['Department']

# 2. Train/test split (same split logic as the Naive Bayes script,
#    same random_state, so results are directly comparable)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization (identical settings to the Naive Bayes script
#    so any difference in results comes from the model, not the features)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train SVM
#    class_weight='balanced' automatically up-weights minority departments
#    (General Inquiry, Human Resources, Sales and Pre-Sales) so the model
#    doesn't just default to predicting the biggest class every time.
svm_model = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
y_pred = svm_model.predict(X_test_tfidf)

print("=== Classification Report (Department) - SVM ===")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Department')
plt.ylabel('Actual Department')
plt.title('Confusion Matrix - SVM (Department)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_department_svm.png', dpi=150)
print("Saved confusion_matrix_department_svm.png")

# 7. Quick manual test with a new made-up ticket
def predict_department(text, vectorizer, model):
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "My laptop screen is completely black and it won't turn on after the update."
print("\nSample prediction:", predict_department(sample_ticket, vectorizer, svm_model))