import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def print_classification_report(y_true, y_pred, model_name, task_name):
    """Prints overall accuracy + full classification report to console."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} ({task_name}) Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"\n=== Classification Report ({task_name}) - {model_name} ===")
    print(classification_report(y_true, y_pred))
    return accuracy


def save_report_metrics(y_true, y_pred, model_name, task_name, root_path):
    """
    Extracts the classification report and saves it as a CSV
    for easy copy-pasting into the assignment report tables.
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_metrics = pd.DataFrame(report_dict).transpose().round(4)

    output_dir = root_path / "report_assets" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name.replace(' ', '_')}_{task_name}_metrics.csv"
    file_path = output_dir / filename
    df_metrics.to_csv(file_path)

    print(f"✅ Saved {model_name} metrics for your report to: {file_path}")
    return file_path


def plot_confusion_matrix(y_true, y_pred, labels, model_name, model_key, task_name,
                           root_path, cmap='Blues', figsize=(10, 8),
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

    output_dir = root_path / "report_assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"confusion_matrix_{task_name.lower()}_{model_key}.png"
    file_path = output_dir / filename
    plt.savefig(file_path, dpi=150)
    plt.close()

    print(f"Saved {filename}")
    return file_path


def save_model_and_vectorizer(model, vectorizer, model_key, task_name, root_path):
    """
    Saves model + vectorizer to prototype/model and prototype/vectorizer,
    using the SAME filenames your app.py already expects
    (e.g. nb_priority_model.pkl, tfidf_nb_priority_vectorizer.pkl).
    """
    model_dir = root_path / "prototype" / "model"
    vec_dir = root_path / "prototype" / "vectorizer"
    model_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    task_short = task_name.lower()
    model_path = model_dir / f"{model_key}_{task_short}_model.pkl"
    vec_path = vec_dir / f"tfidf_{model_key}_{task_short}_vectorizer.pkl"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"Successfully saved {model_path.name} and {vec_path.name}")
    return model_path, vec_path


def load_model_and_vectorizer(model_key, task_name, root_path):
    """Loads a previously saved model + vectorizer pair (mirrors save_model_and_vectorizer)."""
    model_dir = root_path / "prototype" / "model"
    vec_dir = root_path / "prototype" / "vectorizer"

    task_short = task_name.lower()
    model_path = model_dir / f"{model_key}_{task_short}_model.pkl"
    vec_path = vec_dir / f"tfidf_{model_key}_{task_short}_vectorizer.pkl"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer