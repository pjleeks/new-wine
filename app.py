# streamlit_app.py

import streamlit as st
import joblib
import pandas as pd

# Define the file names
VECTORIZER_FILENAME = 'count_vectorizer.pkl'
MODEL_FILENAME = 'logistic_regression_model.pkl'

@st.cache_resource # Caches the resource so it only loads once
def load_assets():
    """Loads the pre-trained model and vectorizer."""
    try:
        # Load the CountVectorizer
        vectorizer = joblib.load(VECTORIZER_FILENAME)
        # Load the trained model
        model = joblib.load(MODEL_FILENAME)
        return vectorizer, model
    except FileNotFoundError:
        st.error(f"Error: Could not find **{VECTORIZER_FILENAME}** or **{MODEL_FILENAME}**. Please ensure the files are in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model/vectorizer: {e}")
        st.stop()

# --- Main Streamlit App ---

st.title("Text Mood Classifier 🎭")
st.markdown("Enter a sentence to predict the underlying emotion.")

# Load the model and vectorizer
vectorizer, model = load_assets()

# Get user input
user_input = st.text_area("Your Text:", "I am so happy that I finally finished the project.")

if st.button("Classify Mood"):
    if user_input:
        # 1. Transform the input text using the loaded vectorizer
        # Note: Must be a list or array-like for the vectorizer
        input_vectorized = vectorizer.transform([user_input])

        # 2. Make the prediction
        prediction = model.predict(input_vectorized)[0]

        # 3. Display the result
        st.success(f"**Predicted Emotion:** **{prediction.upper()}**")

        # Optional: Get probability scores
        # probabilities = model.predict_proba(input_vectorized)[0]
        # prob_df = pd.DataFrame({'Emotion': model.classes_, 'Probability': probabilities})
        # st.subheader("Confidence Scores:")
        # st.dataframe(prob_df.sort_values(by='Probability', ascending=False), use_container_width=True)

    else:
        st.warning("Please enter some text to classify.")
