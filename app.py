import streamlit as st
import joblib
from preprocess import clean_text  # Imports your team's custom text cleaner

# 1. Set up the page design
st.set_page_config(page_title="IT Ticket Triage AI", page_icon="🎫", layout="centered")

st.title("🎫 Automated IT Ticket Triage")
st.write("Enter a customer support ticket below, and our Naive Bayes AI will automatically route it to the correct department and assign a priority level.")

# 2. Load the trained models and vectorizers
# @st.cache_resource ensures the models only load once, making the app much faster
@st.cache_resource
def load_models():
    # Load Department files
    dept_model = joblib.load('nb_department_model.pkl')
    dept_vec = joblib.load('tfidf_department_vectorizer.pkl')
    
    # Load Priority files
    pri_model = joblib.load('nb_priority_model.pkl')
    pri_vec = joblib.load('tfidf_priority_vectorizer.pkl')
    
    return dept_model, dept_vec, pri_model, pri_vec

dept_model, dept_vec, pri_model, pri_vec = load_models()

# 3. Create the user input area
st.subheader("Submit a New Ticket")
ticket_text = st.text_area("Ticket Body:", height=150, placeholder="e.g., URGENT: The main database server is down and no one can process payments!")

# 4. Process the text and make predictions when the user clicks the button
if st.button("Predict Triage Routing"):
    if not ticket_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing text..."):
            # Step A: Preprocess the raw text using your team's pipeline
            cleaned_text = clean_text(ticket_text)
            
            # Step B: Vectorize and predict Department
            dept_vectorized = dept_vec.transform([cleaned_text])
            predicted_dept = dept_model.predict(dept_vectorized)[0]
            
            # Step C: Vectorize and predict Priority
            pri_vectorized = pri_vec.transform([cleaned_text])
            predicted_priority = pri_model.predict(pri_vectorized)[0]
            
            # Step D: Display the Results beautifully
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="🏢 Routed Department", value=predicted_dept)
                
            with col2:
                # Add a little visual flair based on priority level
                if predicted_priority == "High":
                    st.metric(label="🚨 Urgency Priority", value=predicted_priority)
                else:
                    st.metric(label="📋 Urgency Priority", value=predicted_priority)