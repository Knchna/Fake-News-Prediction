# Fake News Classification using NLP

A machine learning-based web application that classifies news articles as **REAL** or **FAKE** using Natural Language Processing (NLP) and Logistic Regression.

The project includes text preprocessing, TF-IDF feature extraction, machine learning classification, model evaluation, and word-level prediction explainability through a Streamlit interface.

---

## Features

* News article classification as **REAL** or **FAKE**
* Prediction confidence and probability scores
* NLP-based text preprocessing
* TF-IDF feature extraction
* Logistic Regression classification
* Comparison with Linear SVM and Random Forest
* Word-level prediction explainability
* Interactive Streamlit web interface

---

## Project Workflow

```text
News Article
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
REAL / FAKE Prediction
     ↓
Confidence & Explainability
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib
* Streamlit

---

## Dataset

The model is trained using two CSV files:

```text
True.csv
Fake.csv
```

The datasets contain real and fake news articles respectively.

### Target Labels

| Label | Meaning |
| ----- | ------- |
| `0`   | REAL    |
| `1`   | FAKE    |

---

## Machine Learning Pipeline

### 1. Text Preprocessing

The news text is processed using:

* Lowercasing
* Punctuation removal
* Number removal
* Tokenization
* Stop-word removal
* Lemmatization

### 2. Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert the processed text into numerical features.

The vectorizer is configured with a maximum of **7,000 features**.

### 3. Models

The following classification models were evaluated:

* Logistic Regression
* Linear SVM
* Random Forest

The deployed application uses **Logistic Regression** because it provides probability estimates and allows word-level coefficient-based explainability.

---

## Project Structure

```text
fake-news-classifier/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── logistic_regression_model.pkl
    ├── tfidf_vectorizer.pkl
    └── model_info.pkl
```

---

# Setup Instructions

## 1. Clone the Repository

Open a terminal or command prompt and run:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd fake-news-classifier
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

If PowerShell is being used:

```powershell
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
streamlit
scikit-learn
pandas
numpy
joblib
nltk
```

---

## 4. Download Required NLTK Resources

The application uses NLTK for tokenization, stop-word removal, and lemmatization.

Run Python:

```bash
python
```

Then execute:

```python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Exit Python:

```python
exit()
```

---

# Running the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in a web browser.

---

# Using the Application

The application contains the following sections:

### Home

Provides an overview of the project and its workflow.

### Predict

1. Enter or paste a news article.
2. Click the prediction button.
3. The application displays:

   * Predicted class
   * Confidence
   * Probability of REAL
   * Probability of FAKE

### Explain

Enter a news article to view the words that contributed most strongly toward:

* **FAKE**
* **REAL**

The explanation is based on the TF-IDF values and Logistic Regression coefficients.

### About

Displays information about:

* Dataset
* Preprocessing
* Model
* Evaluation metrics
* Project methodology

---

# Model Explainability

The Logistic Regression model assigns a coefficient to each TF-IDF feature.

For an individual article:

```text
Contribution = TF-IDF value × Logistic Regression coefficient
```

A positive contribution pushes the prediction toward **FAKE**, while a negative contribution pushes it toward **REAL**.

This provides a simple way to understand which words influenced the model's prediction.

---

# Important Note

This application is a **machine learning classification project**, not a fact-checking system.

The model predicts whether an article resembles the patterns learned from the training dataset. A prediction of **FAKE** does not independently verify that the information is false, and a prediction of **REAL** does not guarantee that the information is factually correct.

---

# Future Improvements

Possible improvements include:

* Improve the Streamlit UI
* Add article input history
* Share the entered article between Prediction and Explainability pages
* Add visualizations for prediction probabilities
* Add additional NLP models
* Experiment with transformer-based models
* Deploy the application online
* Add model performance visualizations
* Improve explainability with more advanced NLP techniques

---

## Author

**Kanchana Krishna**

B.Tech – Computer Science and Engineering (AI & ML)
                  
