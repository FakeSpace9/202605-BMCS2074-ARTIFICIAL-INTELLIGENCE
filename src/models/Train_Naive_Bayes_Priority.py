from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
from pathlib import Path
model_dir = Path(__file__).resolve().parent    # .../src/models
src_dir = model_dir.parent                     # .../src
project_root = src_dir.parent
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import print_classification_report, save_report_metrics, plot_confusion_matrix, save_model_and_vectorizer, load_processed_dataset

sys.path.append(src_dir)

print("Loading data...")
df = load_processed_dataset()
df = df.dropna(subset=['clean_text', 'Priority']) 

X = df['clean_text']
y = df['Priority']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF feature extraction
print("Vectorizing text using word TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=20000,      
    ngram_range=(1, 3),      
    sublinear_tf=True,       
    min_df=2                 
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

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

#Predict and evaluate
y_pred = nb_model.predict(X_test_tfidf)
print_classification_report(y_test, y_pred, "Naive Bayes", "Priority")
save_report_metrics(y_test, y_pred, "Naive Bayes", "Priority")

#Confusion matrix
labels = sorted(y.unique())
plot_confusion_matrix(y_test, y_pred, labels, "Naive Bayes", "nb", "Priority", cmap='Oranges', figsize=(8, 6), xtick_rotation=0)

#Quick manual test
def predict_priority(text, vectorizer, model):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import clean_text
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

sample_ticket = "URGENT: The main database server is down and no one can process payments! Please help immediately."
print("\nSample prediction:", predict_priority(sample_ticket, vectorizer, nb_model))

#Save models
save_model_and_vectorizer(nb_model, vectorizer, "nb", "Priority")
