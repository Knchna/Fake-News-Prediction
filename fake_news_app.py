import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide"
)


# =========================================================
# LOAD NLTK RESOURCES
# =========================================================

@st.cache_resource
def load_nltk_resources():

    try:
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

    except LookupError:

        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


stop_words, lemmatizer = load_nltk_resources()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/logistic_regression_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    model_info = joblib.load(
        "models/model_info.pkl"
    )

    return model, vectorizer, model_info


model, vectorizer, model_info = load_model()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords + lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return processed_tokens


# =========================================================
# PREDICTION
# =========================================================

def predict_news(text):

    processed_tokens = preprocess_text(text)

    cleaned_text = " ".join(processed_tokens)

    text_tfidf = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(
        text_tfidf
    )[0]

    real_probability = probabilities[0]
    fake_probability = probabilities[1]

    confidence = probabilities[prediction]

    label = "FAKE" if prediction == 1 else "REAL"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "tfidf": text_tfidf
    }


# =========================================================
# EXPLAINABILITY
# =========================================================

def explain_prediction(text, top_n=10):

    processed_tokens = preprocess_text(text)

    cleaned_text = " ".join(processed_tokens)

    text_tfidf = vectorizer.transform(
        [cleaned_text]
    )

    feature_names = vectorizer.get_feature_names_out()

    coefficients = model.coef_[0]

    feature_indices = text_tfidf.nonzero()[1]

    contributions = []

    for index in feature_indices:

        word = feature_names[index]

        tfidf_value = text_tfidf[0, index]

        coefficient = coefficients[index]

        contribution = tfidf_value * coefficient

        contributions.append(
            {
                "Word": word,
                "Contribution": contribution
            }
        )

    # Words pushing toward FAKE
    fake_words = sorted(
        [
            x for x in contributions
            if x["Contribution"] > 0
        ],
        key=lambda x: x["Contribution"],
        reverse=True
    )[:top_n]

    # Words pushing toward REAL
    real_words = sorted(
        [
            x for x in contributions
            if x["Contribution"] < 0
        ],
        key=lambda x: x["Contribution"]
    )[:top_n]

    return fake_words, real_words


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📰 Fake News Classifier")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict",
        "Explain",
        "About"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title("📰 Fake News Classification")

    st.subheader(
        "NLP-based Fake News Detection using Machine Learning"
    )

    st.write(
        """
        This application uses Natural Language Processing (NLP)
        and Machine Learning to classify news articles as
        **REAL** or **FAKE**.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "Logistic Regression"
        )

    with col2:
        st.metric(
            "Feature Extraction",
            "TF-IDF"
        )

    with col3:
        st.metric(
            "Task",
            "Binary Classification"
        )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
        **1. News Article**

        User enters a news article.

        ↓

        **2. Text Preprocessing**

        Lowercasing → punctuation removal → number removal →
        tokenization → stopword removal → lemmatization

        ↓

        **3. TF-IDF**

        The cleaned text is converted into numerical features.

        ↓

        **4. Logistic Regression**

        The trained model predicts whether the article is
        **REAL or FAKE**.

        ↓

        **5. Explanation**

        Important words contributing to the prediction are displayed.
        """
    )


# =========================================================
# PREDICT
# =========================================================

elif page == "Predict":

    st.title("🔍 Predict News")

    st.write(
        "Enter a news article below to classify it."
    )

    article = st.text_area(
        "News Article",
        height=300,
        placeholder="Paste the news article here..."
    )

    if st.button(
        "🔎 Predict",
        type="primary"
    ):

        if not article.strip():

            st.warning(
                "Please enter a news article."
            )

        else:

            result = predict_news(article)

            st.divider()

            # Prediction
            if result["label"] == "FAKE":

                st.error(
                    f"🚨 Prediction: {result['label']}"
                )

            else:

                st.success(
                    f"✅ Prediction: {result['label']}"
                )

            # Confidence
            st.subheader("Confidence")

            st.progress(
                result["confidence"]
            )

            st.write(
                f"**{result['confidence']:.2%}**"
            )

            # Probabilities
            st.subheader(
                "Class Probabilities"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🟢 Real",
                    f"{result['real_probability']:.2%}"
                )

            with col2:

                st.metric(
                    "🔴 Fake",
                    f"{result['fake_probability']:.2%}"
                )


# =========================================================
# EXPLAIN
# =========================================================

elif page == "Explain":

    st.title("💡 Explain the Prediction")

    st.write(
        """
        This section shows the words in the article that
        contributed most strongly toward the model's prediction.
        """
    )

    article = st.text_area(
        "News Article",
        height=300,
        placeholder="Paste the same news article here..."
    )

    if st.button(
        "🔎 Analyze Words",
        type="primary"
    ):

        if not article.strip():

            st.warning(
                "Please enter a news article."
            )

        else:

            result = predict_news(article)

            fake_words, real_words = explain_prediction(
                article
            )

            st.subheader(
                f"Prediction: {result['label']}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### 🔴 Words pushing toward FAKE"
                )

                if fake_words:

                    for item in fake_words:

                        st.write(
                            f"**{item['Word']}**  "
                            f"({item['Contribution']:.4f})"
                        )

                else:

                    st.info(
                        "No strong words pushing toward FAKE."
                    )

            with col2:

                st.markdown(
                    "### 🟢 Words pushing toward REAL"
                )

                if real_words:

                    for item in real_words:

                        st.write(
                            f"**{item['Word']}**  "
                            f"({item['Contribution']:.4f})"
                        )

                else:

                    st.info(
                        "No strong words pushing toward REAL."
                    )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title("ℹ️ About the Project")

    st.subheader("Dataset")

    st.write(
        """
        The model was trained using a fake news dataset containing
        separate collections of REAL and FAKE news articles.

        The article title and article text were combined before
        applying NLP preprocessing.
        """
    )

    st.divider()

    st.subheader("NLP Preprocessing")

    st.markdown(
        """
        - Lowercasing
        - Punctuation removal
        - Number removal
        - Tokenization
        - Stopword removal
        - Lemmatization
        """
    )

    st.subheader("Feature Extraction")

    st.write(
        """
        TF-IDF (Term Frequency–Inverse Document Frequency)
        was used to convert text into numerical features.
        """
    )

    st.subheader("Classification Model")

    st.write(
        """
        Logistic Regression was used as the final classification
        model because it works effectively with high-dimensional
        sparse text features and provides class probabilities
        and interpretable coefficients.
        """
    )

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{model_info['accuracy']:.2%}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{model_info['precision']:.2%}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{model_info['recall']:.2%}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{model_info['f1_score']:.2%}"
        )

    st.divider()

    st.caption(
        "Academic NLP / Machine Learning Project"
    )
