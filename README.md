# 🎫 AI-Powered Intelligent Ticket Routing & Resolution Agent

An end-to-end AI system that classifies incoming IT support tickets, routes them to the correct department, retrieves similar past tickets to suggest resolutions, and escalates to human agents when confidence is low — built as a BTech AIML final year project.

> Inspired by the NASSCOM Hackathon Use Case: *"AI Powered Intelligent Ticket Routing & Resolution Agent"*

---

## 📌 Problem Statement

IT services companies receive thousands of support tickets daily, leading to:
- Misrouted tickets sent to the wrong department
- Slow resolution due to manual triaging
- Poor categorization and inconsistent classification
- Repeated issues being solved from scratch instead of reusing past fixes

## 🎯 Project Aim

To design and develop an AI-powered intelligent agent that automatically classifies, routes, and suggests resolutions for IT support tickets — reducing manual triaging effort, minimizing misrouting, and accelerating resolution time through NLP, retrieval-augmented generation (RAG), and confidence-driven escalation.

---

## 🧠 System Architecture
Ticket Submitted
↓
Preprocessing (text cleaning, embedding generation)
↓
Classification Layer → predicts category + confidence score
↓
Routing Layer → maps category to responsible department
↓
RAG Retrieval Layer → finds similar past tickets + resolutions
↓
Decision Layer (LangGraph Agent)
↓ ↓
Auto-Resolve (high confidence) Escalate (low confidence)
↓ ↓
Resolution shown to employee Routed to human agent
↓ ↓
Feedback loop → knowledge base updated


<img width="407" height="586" alt="image" src="https://github.com/user-attachments/assets/e831844d-ccf6-45bd-a6e4-01fe824907d0" />



## 🗂️ Ticket Categories

| Category | Description |
|---|---|
| Infrastructure | Servers, hardware, virtual machines |
| Application | Software bugs, crashes, feature failures |
| Security | Breaches, phishing, unauthorized access |
| Database | Connection failures, slow queries, corruption |
| Storage | Disk space, backup/storage system issues |
| Network | VPN, Wi-Fi/LAN, DNS, connectivity issues |
| Access Management | Password resets, permissions, account lockouts |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Generation | Google Gemini API |
| Classification | Sentence-Transformers (MiniLM) + Logistic Regression / SVM, TF-IDF + Naive Bayes (baseline) |
| Vector Database | ChromaDB (via LangChain) |
| Agent Orchestration | LangGraph |
| Backend API | FastAPI *(planned)* |
| Frontend | Streamlit *(planned)* |
| Database | SQLite / PostgreSQL *(planned)* |
| Version Control | Git + GitHub |

---

## 📊 Dataset

- **994 training tickets** + **119 held-out evaluation tickets**, synthetically generated using Gemini API
- Balanced across all 7 categories (~140 tickets/category)
- Includes augmented edge cases: paraphrased variants, ambiguous cross-category tickets, priority-tone mismatches — designed to realistically test the system's confidence and escalation logic
- Cross-validated against real-world reference datasets (Kaggle IT support ticket datasets) for tone/style realism

## 🤖 Models & Evaluation

| Model | Test Accuracy | Test F1 (macro) |
|---|---|---|
| Baseline: TF-IDF + Naive Bayes | 65.3% | 65.6% |
| **Main: Embeddings (MiniLM) + Logistic Regression** | **70.0%** | **70.3%** |
| Embeddings + SVM | 70.0% | 70.0% |
| 5-Fold Cross-Validation (main model) | 73.4% ± 0.96% | 73.4% ± 1.1% |
| Held-out evaluation set | 67.2% | 67.0% |

**Research contributions:**
- Confidence calibration study revealing the model is *systematically underconfident* rather than overconfident
- Error analysis showing misclassifications concentrate on genuinely ambiguous, conceptually adjacent categories (e.g., Infrastructure ↔ Network) — validating the need for confidence-based escalation

---

## 🔁 Core Features

- ✅ **Automated ticket classification** with confidence scoring
- ✅ **Automatic department routing**
- ✅ **RAG-based resolution suggestion** using semantic similarity search over past resolved tickets
- ✅ **Confidence-aware escalation** — auto-resolves only when both classification confidence and retrieval similarity are high; escalates to a human agent otherwise
- 🔲 **3-way feedback loop** *(planned)* — employee marks a suggestion as Fully Resolved / Partially Resolved / Not Resolved, feeding corrections back into the knowledge base
- 🔲 **Repeated-issue detection** *(planned)* — flags recurring problems as candidates for permanent fixes/automation

---

## 📁 Project Structure
TICKET_BTECH/
├── api/
│ ├── agent.py # LangGraph agentic decision pipeline
│ ├── routing.py # Category → department mapping
│ ├── rag_engine.py # Builds the Chroma vector knowledge base
│ └── rag_retrieval.py # Retrieves similar past tickets
├── data/
│ ├── raw/ # Generated + reference ticket data
│ └── processed/ # Train/val/test splits
├── models/
│ ├── logreg_classifier.joblib
│ ├── baseline_nb_classifier.joblib
│ ├── svm_classifier.joblib
│ └── chroma_db/ # Persisted vector store
├── notebooks/ # Data generation, training, evaluation scripts
├── docs/ # Schema, design notes, evaluation results
├── ui/ # (planned) Frontend
└── requirements.txt

---

## 🔧 Setup & Run

```powershell
cd D:\TICKET_BTECH
.\venv\Scripts\Activate.ps1
pip install setfit datasets
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/app** — employees can choose **LogReg** or **SetFit** before submitting tickets.

---

## 🔧 RAG Knowledge Base

The Chroma vector store is built from **`data/processed/train.csv` only**. Test, validation, and holdout data are never indexed.

```powershell
python api/rag_engine.py
python verify_rag_no_leakage.py
```

---

## 🤖 Train & Compare Classifiers

```powershell
# Baselines
python notebooks/train_baseline_model.py
python notebooks/train_embeddings_model.py
python notebooks/train_svm_model.py

# SetFit (contrastive fine-tuning — ~60–75 min on CPU)
python notebooks/train_setfit_model.py

# Compare all models
python notebooks/compare_models.py

# Held-out generalization
python notebooks/evaluate_holdout.py
python notebooks/evaluate_setfit_holdout.py

# Single ticket smoke test (both models)
python test_single_ticket.py
python test_single_ticket.py --model setfit
```

**Current test accuracy (~695 train samples):** LogReg ~70% | SetFit ~71%

---

## 📦 Scale Dataset to 4k–5k (optional)

1. Save LLM batches as pipe-delimited files in `data/raw/llm_batches/`
2. Merge: `python notebooks/Data_Processing/merge_llm_batches.py`
3. Validate: `python notebooks/Data_Processing/validate_v2_pool.py`
4. Split: `python notebooks/Data_Processing/split_data.py --input data/raw/all_tickets_full_v2.json`
5. Rebuild Chroma + retrain SetFit

---

## 🚀 How It Works — Example :
Input: "My VPN keeps disconnecting every few minutes on Windows 11, error 807"

→ Classified as: Network (confidence: 0.52)
→ Routed to: Network Operations Team
→ Retrieved similar past ticket (similarity: 0.53):
"VPN connection failing with Error 807..."
→ Decision: AUTO-RESOLVE
→ Suggested Resolution: "Identified an expired security certificate on the
VPN gateway. Renewed the SSL certificate and re-provisioned the tunnel
configuration."

---

## 📈 Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Dataset generation & validation | ✅ Complete |
| 2 | Classifier training & evaluation | ✅ Complete |
| 3 | Routing logic | ✅ Complete |
| 4 | RAG resolution layer | ✅ Complete |
| 5 | Agentic decision layer (LangGraph) | ✅ Complete |
| 6 | Backend API (FastAPI) | ✅ Complete |
| 7 | Frontend UI | ✅ Complete |
| 8 | Dual classifier (LogReg + SetFit) | ✅ Complete |
| 9 | Dataset scale-up (4k–5k) | 🔲 In Progress |
| 10 | Deployment | 🔲 Planned |

---

## 🎓 Academic Context

Developed as a BTech AIML final year project, based on the NASSCOM Hackathon Use Case 1: *"AI Powered Intelligent Ticket Routing & Resolution Agent."* Positioned within the broader **AIOps (AI for IT Operations)** industry category, addressing real Mean-Time-To-Resolution (MTTR) and support-cost challenges faced by IT services companies.

## 👤 Author

Developed by Shreyansh Pandey — Symbiosis Institute of Technology (SIT)  
Department of Artificial Intelligence and Machine Learning

---

## 📄 License

This project is for academic purposes as part of a BTech final year project.
