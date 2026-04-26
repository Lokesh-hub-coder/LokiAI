# LokiAI — Local AI + VectorDB (Python)

A fully working **local AI system** built using **Python (FastAPI)** with a custom **Vector Database + RAG pipeline**.

Runs completely on your machine using **Ollama** — no cloud required.

---

## 🚀 Features

* ⚡ FastAPI backend (Python)
* 🧠 Local LLM (Ollama)
* 📦 Vector Search (HNSW + KDTree + BruteForce)
* 🔍 Semantic Search (Cosine / Euclidean / Manhattan)
* 📄 Document Embedding (RAG)
* 🤖 Ask AI from your own data
* 📊 Interactive UI (PCA visualization + chat)

---

## 🧠 How It Works

```
Your Question
     ↓
Embedding (Ollama)
     ↓
Vector Search (HNSW)
     ↓
Top-K Context
     ↓
LLM (llama3)
     ↓
Answer
```

---

## ⚙️ Requirements

* Python 3.9+
* Ollama installed

Install Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## ▶️ Run the Project

### Step 1: Install dependencies

```bash
pip install fastapi uvicorn httpx
```

---

### Step 2: Start backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8080
```

---

### Step 3: Open in browser

```
http://localhost:8080
```

---

## 🧪 Usage

### 🔍 Search

* Enter query → get similar vectors
* Choose algorithm (HNSW / KDTree / Brute)

### 📄 Documents (RAG)

* Insert text
* Auto embedding via Ollama

### 🤖 Ask AI

* Ask questions from your documents
* Adjust **Top-K** (context size)

---

## ⚠️ Important Notes

* Data is stored in memory (not persistent yet)
* Restart = data reset
* Ollama must be running

---

## 🧠 Tech Stack

* FastAPI (backend)
* HTML + JS (frontend)
* Ollama (LLM + embeddings)
* Custom VectorDB

---

## 🚀 Future Improvements

* Persistent storage (save/load vectors)
* Chat history memory
* Deployment (public access)
* UI enhancements

---

## 📌 Author

Built as a learning + real AI system project 🚀
