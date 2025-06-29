# 🧠 Full-Stack ML + RAG Chatbot App

A full-stack Streamlit application that combines **real-time machine learning predictions** with a **RAG (Retrieval-Augmented Generation) chatbot** powered by **Groq's LLaMA models** and local documentation.

---

## 📌 Project Overview

This project simulates live customer interaction data and performs:
- **Real-time customer value prediction** using a trained ML model.
- **Conversational Q&A** over local documentation (Python/ML/SQL PDFs) using FAISS + HuggingFace embeddings + LangChain + Groq LLM.

The app has **two main features**:
- **📊 Real-Time Dashboard**: Predicts whether a simulated customer has *high* or *low* value using user behavior.
- **💬 RAG Chatbot**: Allows users to query documentation using a Groq-powered LLM backed by a local FAISS index.

---

## ⚙️ Setup & Execution Instructions


### ✅ 1. Install dependencies

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

**`requirements.txt` includes**:
- streamlit, scikit-learn, pandas, numpy, matplotlib
- sentence-transformers, langchain, faiss-cpu, PyMuPDF
- langchain-groq, joblib, imbalanced-learn

### ✅ 2. Place your PDF documents

Add your documentation files (e.g. Python, ML, SQL) inside the `data/` folder.

```
project-root/
├── data/
│   └── your_docs.pdf
```

### ✅ 3. Set your Groq API key

You can set it via environment variable:

```bash
export GROQ_API_KEY="your-key"    # On Linux/macOS
set GROQ_API_KEY=your-key         # On Windows
```

Or edit the hardcoded key inside `rag_chatbot.py`.

### ✅ 4. Run the app

```bash
streamlit run app.py
```

---

## 🧠 Model & Tool Explanation

### 📈 Machine Learning Model

- **Model Type**: `RandomForestClassifier` (scikit-learn)
- **Training Data**: Simulated customer behavior including:
  - Product type, price, number of clicks
  - Device type, region, age, session time
  - Whether the user is returning
- **Target**: Binary classification – "high" vs "low" customer value
- **Preprocessing**:
  - One-hot encoding for categorical variables
  - Resampling via `RandomOverSampler` to handle class imbalance
- **Saved Model**: `resampled_customer_value_model.joblib`

### 🤖 RAG Chatbot (LangChain + Groq + FAISS)

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Text Chunking**: `RecursiveCharacterTextSplitter` with 500-char chunks
- **Vector Store**: `FAISS` (stored in `faiss_index/`)
- **LLM**: Groq-hosted LLaMA 3.3-70B (`llama-3.3-70b-versatile`) via `langchain-groq`
- **Pipeline**:
  - Load PDFs using `PyMuPDF`
  - Split into chunks and embed
  - Store in FAISS and retrieve top-k relevant chunks
  - Generate answer using LLM

---

## 🗂️ Project Structure

```
├── app.py                      # Main Streamlit interface
├── rag_chatbot.py             # RAG chatbot logic
├── model_training.py          # (your training code)
├── resampled_customer_value_model.joblib
├── realtime_sales.csv         # Simulated dataset
├── faiss_index/               # Auto-generated vector index
├── data/                      # Folder for documentation PDFs
├── requirements.txt
└── README.md
```

---

## 📈 Visualization Example

- Live data appears in a growing table
- Bar chart shows number of high/low value predictions over time
- Sidebar controls for simulation speed, filters, and row limits

---

## 💡 Use Cases

- Teaching/learning full-stack ML deployment
- Exploring LangChain-based RAG pipelines
- Building fast LLM apps with Groq
- Simulating real-time data dashboards

---

## 📄 License

MIT License — Free to use, modify, and share.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [HuggingFace](https://huggingface.co/)
