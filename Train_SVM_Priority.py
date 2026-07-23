"""
SVM Model to Classify Ticket Priority
Reads from 'cleaned_tickets.csv', extracts TF-IDF features,
trains a Support Vector Machine, and evaluates the results.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load cleaned data
print("Loading cleaned dataset...")
df = pd.read_csv('cleaned_tickets.csv')
# Drop any rows where the text or priority might be missing
df = df.dropna(subset=['clean_text', 'Priority'])

X = df['clean_text']
y = df['Priority']


# 2. Train/Test Split (80% training, 20% testing)
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
print("Extracting features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=350000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Initialize and Train the SVM
print("Training the SVM model for Priority (this might take a moment)...")
svm_model = LinearSVC(C = 4.4, class_weight='balanced', max_iter=450, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# 5. Predict and evaluate
print("Predicting & Evaluating the model...")
y_pred = svm_model.predict(X_test_tfidf)

print("=== Classification Report (Priority) - SVM ===")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix
print("Generating Confusion Matrix visualization...")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
# Using 'Blues' instead of 'Greens' so you can tell the priority and dept charts apart
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Priority')
plt.ylabel('Actual Priority')
plt.title('Confusion Matrix - SVM (Priority)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_priority_svm.png', dpi=150)
print("Confusion matrix saved as 'confusion_matrix_priority_svm.png'!")

# 7. Quick Manual Test
def predict_priority(text, vectorizer, model):
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# sample_ticket = "The entire main server is completely offline and no one can work!"
sample_ticket = "Can someone help me configure the SAML integration settings for the guest Wi-Fi? No rush."
# sample_ticket = "Massive broadcast storm on the Omada controller, the entire network profile is locked up and switches are dropping."
print("\nSample prediction:", predict_priority(sample_ticket, vectorizer, svm_model))

# 8. Save the model and vectorizer to disk
print("\nSaving models to disk...")
joblib.dump(svm_model, 'svm_priority_model.pkl')
joblib.dump(vectorizer, 'tfidf_svm_priority_vectorizer.pkl')
print("Successfully saved svm_priority_model.pkl and tfidf_svm_priority_vectorizer.pkl")
