#  Claims Intelligence Copilot

A highly specialized, secure, and offline-first **Retrieval-Augmented Generation (RAG)** application designed for the insurance and financial services sectors. The Claims Intelligence Copilot allows claims adjusters, underwriters, and compliance officers to upload massive policy PDFs and ask natural language questions about coverage, exclusions, and claim eligibility with absolute precision.

##  Why is this unique? (The Market Gap)

Most enterprise solutions today rely on uploading highly confidential documents directly to third-party LLMs like ChatGPT or Claude. This poses massive data privacy, security, and compliance risks. Furthermore, standard LLMs are prone to "hallucinations," making them unreliable for legal or claims-based decisions where exact wording is paramount.

**The Claims Intelligence Copilot solves this by:**
1. **100% Local Embedding:** Documents are processed, chunked, and embedded entirely locally on your machine. Your proprietary PDF data never leaves your infrastructure during the indexing phase.
2. **Zero Hallucinations & Verifiable Citations:** The LLM is strictly prompted to answer *only* based on the retrieved text. Every answer is accompanied by an expandable citation panel showing the exact PDF filename, page number, and original text chunk used to generate the response.
3. **Algorithmic Confidence Scoring:** Unlike standard chatbots, this system mathematically evaluates the "cosine similarity" of your query against the document database. It pre-evaluates its own findings with a `HIGH`, `MEDIUM`, or `REVIEW REQUIRED` confidence badge *before* generating text.

## 🛠 Tech Stack

- **Frontend Interface:** [Streamlit](https://streamlit.io/) (Custom styled for a premium, dark-mode copilot experience)
- **Orchestration Framework:** [LlamaIndex](https://www.llamaindex.ai/) (For seamless data ingestion, RAG pipeline construction, and query routing)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Running locally for persistent vector storage)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (Open-source, free, and runs entirely locally)
- **Large Language Model:** `Llama-3.1-8B-instant` via the [Groq](https://groq.com/) API (Lightning-fast inference speeds)

## ⚙️ How It Works Under the Hood

1. **Ingestion (`ingest.py` / Streamlit UI):**
   - The user uploads a 300+ page PDF.
   - `SimpleDirectoryReader` extracts the text.
   - `SentenceSplitter` breaks the massive document into dense, overlapping 512-token chunks.
   - The HuggingFace embedding model converts these text chunks into numerical vectors.
   - These vectors, alongside metadata (page numbers, filenames), are saved into the local `ChromaDB`.

2. **Querying (`app.py`):**
   - The user asks a question (e.g., *"Is water damage from a burst pipe covered?"*).
   - The system embeds the query and searches the `ChromaDB` for the top 5 most mathematically similar text chunks across all uploaded policies.
   - These 5 chunks are injected into a highly restrictive System Prompt.
   - Groq's Llama-3.1 model reads the prompt and the chunks, synthesizes a human-readable answer, and the UI displays the result alongside the confidence score and source citations.

## Getting Started

### Prerequisites
- Python 3.9+
- A free [Groq API Key](https://console.groq.com/keys)

### Installation

1. **Clone the repository and navigate to the directory:**
   ```bash
   cd /path/to/financial_document
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API Key:**
   - Create a `.env` file in the root directory (or use `.env.example` as a template).
   - Add your API key:
     ```env
     GROQ_API_KEY=gsk_your_api_key_here
     ```
   *(Alternatively, you can input your API key directly into the Streamlit sidebar).*

### Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`. 

From there:
1. Upload your policy PDFs in the sidebar.
2. Click **Process & Index Documents**.
3. Begin chatting with your policy data!
# rag-based-application-
