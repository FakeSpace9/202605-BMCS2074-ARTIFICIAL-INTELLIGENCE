import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

model_dir = Path(__file__).resolve().parent    # .../src/models
src_dir = model_dir.parent                     # .../src
project_root = src_dir.parent

def print_classification_report(y_true, y_pred, model_name, task_name):
    """Prints overall accuracy + full classification report to console."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} ({task_name}) Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"\n=== Classification Report ({task_name}) - {model_name} ===")
    print(classification_report(y_true, y_pred))
    return accuracy


def save_report_metrics(y_true, y_pred, model_name, task_name):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_metrics = pd.DataFrame(report_dict).transpose().round(4)


    output_dir = project_root / "report_assets" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)


    filename = f"{model_name.replace(' ', '_')}_{task_name}_metrics.csv"
    

    file_path = output_dir / filename

    df_metrics.to_csv(file_path)

    print(f"✅ Saved {model_name} metrics for your report to: {file_path}")
    return file_path

def plot_confusion_matrix(y_true, y_pred, labels, model_name, model_key, task_name,
                        cmap='Blues', figsize=(10, 8),
                           xtick_rotation=45, xtick_ha='right'):
    """
    Builds a confusion-matrix heatmap and saves it to report_assets/plots.
    model_key is the short filename code (e.g. 'nb', 'svm', 'logreg') so
    filenames stay consistent with the existing *_model.pkl naming.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel(f'Predicted {task_name}')
    plt.ylabel(f'Actual {task_name}')
    plt.title(f'Confusion Matrix - {model_name} ({task_name})')
    plt.xticks(rotation=xtick_rotation, ha=xtick_ha if xtick_rotation else 'center')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_dir = project_root / "report_assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"confusion_matrix_{task_name.lower()}_{model_key}.png"
    file_path = output_dir / filename
    plt.savefig(file_path, dpi=150)
    plt.close()

    print(f"Saved {filename}")
    return file_path

def get_prediction_confidence(model, vectorized_text):
    """
    Predict a class and return a confidence/score.
    For models with predict_proba():
        Uses the probability of the predicted class.
    For models without predict_proba():
        Uses decision_function() and converts the scores into
        a relative confidence using softmax.
    Returns:
        predicted_class
        confidence
        probabilities
    """

    predicted_class = model.predict(vectorized_text)[0]

    # 1. Models with predict_proba()
    #    - MultinomialNB
    #    - LogisticRegression
    #    - SVC(probability=True)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_text)[0]
        class_labels = model.classes_

        predicted_index = list(class_labels).index(predicted_class)
        confidence = probabilities[predicted_index]

        probabilities_dict = {
            label: float(probability)
            for label, probability in zip(class_labels, probabilities)
        }

        return predicted_class, confidence, probabilities_dict


    # 2. Models without predict_proba()
    #    Example:
    #    - LinearSVC
    if hasattr(model, "decision_function"):
        scores = model.decision_function(vectorized_text)
        class_labels = model.classes_

        # Binary classification
        if scores.ndim == 1:
            score = float(scores[0])

            # Convert binary decision score to a probability-like value
            import math

            confidence = 1 / (1 + math.exp(-abs(score)))

            # Assign scores to the two classes
            if score >= 0:
                probabilities_dict = {
                    class_labels[0]: 1 - confidence,
                    class_labels[1]: confidence
                }
            else:
                probabilities_dict = {
                    class_labels[0]: confidence,
                    class_labels[1]: 1 - confidence
                }

        # Multiclass classification
        else:
            import numpy as np

            scores = scores[0]

            # Softmax
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()

            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])

            probabilities_dict = {
                label: float(probability)
                for label, probability in zip(class_labels, probabilities)
            }

        return predicted_class, confidence, probabilities_dict
    return predicted_class, None, None
def save_model_and_vectorizer(model, vectorizer, model_key, task_name):
    model_dir = project_root / "prototype" / "model"
    vec_dir = project_root / "prototype" / "vectorizer"
    model_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    task_short = task_name.lower()
    model_path = model_dir / f"{model_key}_{task_short}_model.pkl"
    vec_path = vec_dir / f"tfidf_{model_key}_{task_short}_vectorizer.pkl"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"Successfully saved {model_path.name} and {vec_path.name}")
    return model_path, vec_path


def load_model_and_vectorizer(model_key, task_name):
    """Loads a previously saved model + vectorizer pair (mirrors save_model_and_vectorizer)."""
    model_dir = project_root / "prototype" / "model"
    vec_dir = project_root / "prototype" / "vectorizer"

    task_short = task_name.lower()
    model_path = model_dir / f"{model_key}_{task_short}_model.pkl"
    vec_path = vec_dir / f"tfidf_{model_key}_{task_short}_vectorizer.pkl"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def load_processed_dataset():
    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "data" / "processed" / "cleaned_tickets.csv"
    
    df = pd.read_csv(file_path)
    return df