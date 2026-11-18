import streamlit as st
import pandas as pd
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# DOWNLOAD NLTK RESOURCES
# -----------------------------
@st.cache_resource
def load_nltk_resources():
    resources = ["punkt", "punkt_tab", "wordnet", "stopwords"]

    for resource in resources:
        nltk.download(resource, quiet=True)

    return (
        WordNetLemmatizer(),
        set(stopwords.words("english"))
    )


lemmatizer, stop_words = load_nltk_resources()


# -----------------------------
# ORIGINAL TRAINING DATA
# -----------------------------
data = {
    "review": [
        "The movie was great",
        "The movie was okay",
        "The movie was terrible",
        "I loved the movie",
        "I hated the movie"
    ],
    "sentiment": [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())

    processed_text = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(processed_text)


# -----------------------------
# MODEL TRAINING
# -----------------------------
@st.cache_resource
def train_model():

    df["processed"] = df["review"].apply(preprocess)

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df["processed"])
    y = df["sentiment"]

    model = LogisticRegression()

    model.fit(X, y)

    return model, vectorizer


model, vectorizer = train_model()


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and let the Machine Learning "
    "model predict whether the sentiment is Positive or Negative."
)

st.divider()

# User input
user_review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: The movie was absolutely amazing!",
    height=150
)

# Prediction button
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if user_review.strip():

        # Preprocess input
        processed_review = preprocess(user_review)

        # Convert text into numerical features
        vectorized_review = vectorizer.transform(
            [processed_review]
        )

        # Model prediction
        prediction = model.predict(vectorized_review)[0]

        # Prediction probability
        probabilities = model.predict_proba(
            vectorized_review
        )[0]

        positive_probability = probabilities[1]
        negative_probability = probabilities[0]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("😊 Positive Sentiment")

        else:

            st.error("😞 Negative Sentiment")

        # Display probabilities
        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Positive",
                f"{positive_probability * 100:.2f}%"
            )

            st.progress(float(positive_probability))

        with col2:
            st.metric(
                "Negative",
                f"{negative_probability * 100:.2f}%"
            )

            st.progress(float(negative_probability))

        # Processed text
        with st.expander("View Preprocessed Text"):
            st.write(processed_review)

    else:

        st.warning("Please enter a movie review first.")


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("About the Model")

st.sidebar.write("""
**Algorithm:** Logistic Regression

**Feature Extraction:** CountVectorizer

**NLP:** NLTK

**Preprocessing:**
- Tokenization
- Stopword Removal
- Lemmatization

**Task:** Binary Sentiment Classification
""")

st.sidebar.divider()

st.sidebar.caption(
    "Developed using Python, Scikit-learn and Streamlit."
)


