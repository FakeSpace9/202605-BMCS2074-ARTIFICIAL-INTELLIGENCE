from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import sys
from pathlib import Path

model_dir = Path(__file__).resolve().parent    # .../src/models
src_dir = model_dir.parent                     # .../src
project_root = src_dir.parent
sys.path.append(src_dir)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import print_classification_report, save_report_metrics, plot_confusion_matrix, save_model_and_vectorizer, load_processed_dataset

# 1. Load the cleaned data.  
print("loading data...")
df = load_processed_dataset()
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
    max_features=48000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True,
    min_df=2
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train ComplementNB with alpha tuning via GridSearchCV
# MultinomialNB makes on imbalanced datasets — no manual sample_weight needed.
param_grid = {
    'alpha': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
    'norm': [False, True],
}
grid = GridSearchCV(
    ComplementNB(),
    param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid.fit(X_train_tfidf, y_train)

nb_model = grid.best_estimator_


print(f"ComplementNB best CV accuracy: {grid.best_score_:.4f} ({grid.best_params_})")

print(f"CV accuracy: {grid.best_score_:.4f}")
print("\nClass distribution:")
print(y.value_counts())

#Predict and evaluate
y_pred = nb_model.predict(X_test_tfidf)
print_classification_report(y_test, y_pred, "Naive Bayes", "Department")
save_report_metrics(y_test, y_pred, "Naive Bayes", "Department")

#Confusion matrix
labels = sorted(y.unique())
plot_confusion_matrix(y_test, y_pred, labels, "Naive Bayes", "nb", "Department", cmap='Blues', figsize=(8, 6), xtick_rotation=0)


#Quick manual test
def predict_department(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "My laptop screen is completely black and it won't turn on after the update."
print("\nSample prediction:", predict_department(sample_ticket, vectorizer, nb_model))

#Save models
save_model_and_vectorizer(nb_model, vectorizer, "nb", "Department")
