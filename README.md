# 🛍️ ProdReco - Product Recommendation System

**ProdReco** is an intelligent product recommendation system built using product data from Flipkart. It leverages Natural Language Processing (NLP) and vector-based similarity techniques to suggest the top 5 most relevant products similar to a given search query. The entire project is deployed as an interactive **Streamlit** app.

---

## 🚀 Features

- Cleaned and preprocessed product descriptions using NLP techniques
- Built a corpus using various vectorization methods: `TF-IDF`, `BoW`, `Word2Vec`, `GloVe`, `FastText`
- Identified **Word2Vec** as the best-performing embedding technique for this dataset
- Generated a **cosine similarity matrix** to compare product descriptions
- Returns **top 5 product recommendations** with similarity scores
- Fully functional and easy-to-use **Streamlit UI**

---

## 🔍 Tech Stack

- **Language**: Python 3.x  
- **NLP Libraries**: NLTK, SpaCy, Gensim  
- **Vectorization**: Scikit-learn, Gensim  
- **Web App**: Streamlit  
- **Data**: Flipkart Product Dataset (CSV)

---

## 🧠 Methodology

1. **Data Preprocessing**
    - Removal of stopwords, punctuation, special characters, and symbols
    - Applied stemming and lemmatization for normalization
    - Cleaned product titles and descriptions

2. **Corpus Creation**
    - Constructed a text corpus from cleaned product descriptions
    - Tokenized and embedded the corpus using various vectorization techniques

3. **Similarity Computation**
    - Applied `cosine similarity` on the embedded corpus
    - Built a similarity matrix to identify the top 5 similar products

4. **Recommendation Engine**
    - User inputs a product name or keyword
    - App returns 5 closest matching products along with similarity scores

---

## 📸 Screenshot

 <!--    ![App Screenshot](screenshot.png) Replace with actual screenshot path    -->

---

## 🧪 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ProdReco.git
   cd ProdReco

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py