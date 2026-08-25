import streamlit as st
import joblib
from pathlib import Path

from preprocess import clean_text
from utils import get_prediction_confidence


def validate_ticket_input(text):
    """Validates ticket length."""
    raw_words = text.split()

    if len(raw_words) < 5:
        return False, "Ticket is too short. Please provide at least 5 words."

    if len(raw_words) >= 500:
        return False, "Ticket is too long. Please summarize the issue."

    return True, "Valid"


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IT Ticket Triage AI",
    page_icon="🎫",
    layout="centered"
)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "prototype" / "model"
VEC_DIR = BASE_DIR / "prototype" / "vectorizer"


# ============================================================
# 3. MODEL DICTIONARIES
# ============================================================

DEPT_MODELS = {
    "Naive Bayes": (
        MODEL_DIR / "nb_department_model.pkl",
        VEC_DIR / "tfidf_nb_department_vectorizer.pkl"
    ),

    "SVM": (
        MODEL_DIR / "svm_department_model.pkl",
        VEC_DIR / "tfidf_svm_department_vectorizer.pkl"
    ),

    "Logistic Regression": (
        MODEL_DIR / "logreg_department_model.pkl",
        VEC_DIR / "tfidf_logreg_department_vectorizer.pkl"
    )
}


PRIORITY_MODELS = {
    "Naive Bayes": (
        MODEL_DIR / "nb_priority_model.pkl",
        VEC_DIR / "tfidf_nb_priority_vectorizer.pkl"
    ),

    "SVM": (
        MODEL_DIR / "svm_priority_model.pkl",
        VEC_DIR / "tfidf_svm_priority_vectorizer.pkl"
    ),

    "Logistic Regression": (
        MODEL_DIR / "logreg_priority_model.pkl",
        VEC_DIR / "tfidf_logreg_priority_vectorizer.pkl"
    )
}


# ============================================================
# 4. MODEL LOADER
# ============================================================

@st.cache_resource
def load_selected_model(model_path, vec_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    return model, vectorizer


# ============================================================
# 5. PAGE HEADER
# ============================================================

st.title("🎫 Automated IT Ticket Triage")

st.write(
    "Enter a customer support ticket below, and our AI will "
    "automatically route it to the correct department and assign "
    "a priority level."
)


# ============================================================
# 6. SIDEBAR MODEL SELECTION
# ============================================================

st.sidebar.header("⚙️ Settings")

st.sidebar.write(
    "Select the underlying AI models for the triage system:"
)


selected_dept_model = st.sidebar.selectbox(
    "Department Routing Model",
    list(DEPT_MODELS.keys())
)


selected_pri_model = st.sidebar.selectbox(
    "Priority Prediction Model",
    list(PRIORITY_MODELS.keys())
)


# ============================================================
# 7. GET MODEL FILE PATHS
# ============================================================

dept_model_file, dept_vec_file = DEPT_MODELS[
    selected_dept_model
]

pri_model_file, pri_vec_file = PRIORITY_MODELS[
    selected_pri_model
]


# ============================================================
# 8. LOAD MODELS
# ============================================================

try:

    dept_model, dept_vec = load_selected_model(
        dept_model_file,
        dept_vec_file
    )

    pri_model, pri_vec = load_selected_model(
        pri_model_file,
        pri_vec_file
    )

except FileNotFoundError as e:

    st.error(
        f"⚠️ Missing file: {e}. "
        f"Make sure you have trained and saved the "
        f"{selected_dept_model} and {selected_pri_model} "
        f"models to your folder."
    )

    st.stop()


# ============================================================
# 9. USER INPUT
# ============================================================

st.subheader("Submit a New Ticket")

ticket_text = st.text_area(
    "Ticket Body:",
    height=150,
    placeholder=(
        "e.g., URGENT: The main database server is down "
        "and no one can process payments! (5~500 words)"
    )
)


# ============================================================
# 10. PREDICTION BUTTON
# ============================================================

if st.button("Predict Triage Routing"):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    is_valid, validation_message = validate_ticket_input(
        ticket_text
    )

    if not is_valid:

        st.warning(
            f"⚠️ {validation_message}"
        )

    else:

        with st.spinner("Analyzing text..."):

            # =================================================
            # STEP A: PREPROCESS TEXT
            # =================================================

            cleaned_text = clean_text(
                ticket_text,
                remove_custom_stopwords=True
            )

            # =================================================
            # STEP B: DEPARTMENT PREDICTION
            # =================================================

            dept_vectorized = dept_vec.transform(
                [cleaned_text]
            )

            (
                predicted_dept,
                dept_confidence,
                dept_probabilities
            ) = get_prediction_confidence(
                dept_model,
                dept_vectorized
            )

            # =================================================
            # STEP C: PRIORITY PREDICTION
            # =================================================

            pri_vectorized = pri_vec.transform(
                [cleaned_text]
            )

            (
                predicted_priority,
                pri_confidence,
                pri_probabilities
            ) = get_prediction_confidence(
                pri_model,
                pri_vectorized
            )

            # =================================================
            # STEP D: STREAMLIT RESULT DISPLAY
            # =================================================

            st.success("Analysis Complete!")

            col1, col2 = st.columns(2)

            # =================================================
            # DEPARTMENT RESULT
            # =================================================

            with col1:

                st.metric(
                    label=(
                        f"🏢 Routed Department "
                        f"({selected_dept_model})"
                    ),
                    value=predicted_dept
                )


                if dept_confidence is not None:

                    st.write(
                        f"**Confidence: "
                        f"{dept_confidence:.1%}**"
                    )

                    st.progress(
                        float(dept_confidence)
                    )

            # =================================================
            # PRIORITY RESULT
            # =================================================

            with col2:

                if predicted_priority.lower() == "high":

                    st.metric(
                        label=(
                            f"🚨 Urgency Priority "
                            f"({selected_pri_model})"
                        ),
                        value=predicted_priority
                    )

                else:

                    st.metric(
                        label=(
                            f"📋 Urgency Priority "
                            f"({selected_pri_model})"
                        ),
                        value=predicted_priority
                    )


                if pri_confidence is not None:

                    st.write(
                        f"**Confidence: "
                        f"{pri_confidence:.1%}**"
                    )

                    st.progress(
                        float(pri_confidence)
                    )

            # =================================================
            # OPTIONAL: SHOW ALL PROBABILITIES
            # =================================================

            st.divider()

            st.subheader("📊 Prediction Details")

            detail_col1, detail_col2 = st.columns(2)

            # -------------------------------------------------
            # Department probabilities
            # -------------------------------------------------

            with detail_col1:

                st.write(
                    f"**Department probabilities "
                    f"({selected_dept_model})**"
                )

                if dept_probabilities:

                    for label, probability in sorted(
                        dept_probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):

                        st.write(
                            f"{label}: {probability:.1%}"
                        )

                        st.progress(
                            float(probability)
                        )

                else:

                    st.info(
                        "This model does not provide "
                        "probability scores."
                    )

            # -------------------------------------------------
            # Priority probabilities
            # -------------------------------------------------

            with detail_col2:

                st.write(
                    f"**Priority probabilities "
                    f"({selected_pri_model})**"
                )

                if pri_probabilities:

                    for label, probability in sorted(
                        pri_probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):

                        st.write(
                            f"{label}: {probability:.1%}"
                        )

                        st.progress(
                            float(probability)
                        )

                else:

                    st.info(
                        "This model does not provide "
                        "probability scores."
                    )