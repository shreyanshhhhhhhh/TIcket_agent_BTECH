"""
Generate CA-3 Second Presentation PPT from institute template.
Run: python notebooks/generate_ca3_presentation.py
"""
import os
from pptx import Presentation
from pptx.util import Pt

TEMPLATE = r"c:\Users\HP\OneDrive\Downloads\B_Tech_Project_CA-3_Template_Second_Presentation.pptx"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE_DIR, "docs", "B_Tech_Project_CA3_Second_Presentation.pptx")
OUTPUT_COPY = r"c:\Users\HP\OneDrive\Downloads\B_Tech_Project_CA3_Second_Presentation.pptx"

OBJECT_LAYOUT = 1  # Title + Content
SECTION_LAYOUT = 2


def set_title(slide, text: str):
    if slide.shapes.title:
        slide.shapes.title.text = text


def set_body(slide, lines: list[str], font_size=18):
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        if p.runs:
            p.runs[0].font.size = Pt(font_size)
            p.runs[0].font.name = "Times New Roman"


def add_content_slide(prs, title: str, lines: list[str], font_size=18):
    slide = prs.slides.add_slide(prs.slide_layouts[OBJECT_LAYOUT])
    set_title(slide, title)
    set_body(slide, lines, font_size=font_size)
    return slide


def main():
    prs = Presentation(TEMPLATE)

    # --- Slide 1: Title (update template placeholders) ---
    s1 = prs.slides[0]
    for shape in s1.shapes:
        if not shape.has_text_frame:
            continue
        t = shape.text
        if "Title of the Project" in t:
            shape.text = "Title of the Project: AI-Powered Intelligent Ticket Routing & Resolution Agent"
        elif t.startswith("Group Members"):
            shape.text = (
                "Group Members:\n"
                "Shreyansh Pandey (PRN: ____________)\n"
                "Name (PRN: ____________)\n"
                "Name (PRN: ____________)\n"
                "Name (PRN: ____________)"
            )
        elif "Name of the Guide" in t and "Co-guide" not in t:
            shape.text = "Name of the Guide: ________________"
        elif "Co-guide" in t:
            shape.text = "Name of the Co-guide: ______________"
        elif "Project ID" in t:
            shape.text = "Project ID: __________________"

    # --- Slide 2: Outline (already in template) ---

    # --- Slide 3: Use empty template slide for architecture overview ---
    if len(prs.slides) >= 3:
        s3 = prs.slides[2]
        if s3.shapes.title:
            s3.shapes.title.text = "System Architecture Overview"
        if len(s3.placeholders) > 1:
            set_body(
                s3,
                [
                    "Ticket Submitted (Employee Portal)",
                    "  ↓  Preprocessing & Embedding (MiniLM-L6-v2)",
                    "  ↓  Classification (LogReg / SetFit) + Confidence",
                    "  ↓  Routing (Category → Department)",
                    "  ↓  RAG Retrieval (ChromaDB — train tickets only)",
                    "  ↓  LangGraph Decision Agent",
                    "  ↓  Auto-Resolve (high confidence)  |  Escalate (low confidence)",
                    "  ↓  Resolution shown to user          |  Agent queue for human review",
                ],
                font_size=16,
            )

    # --- Content slides per CA-3 outline ---
    slides_data = [
        (
            "Group Members & Work Distribution",
            [
                "Shreyansh Pandey — Project lead, ML pipeline, API & frontend integration",
                "Member 2 — Dataset generation, data preprocessing & validation",
                "Member 3 — Model training, evaluation & documentation",
                "Member 4 — Testing, UI development & deployment",
                "",
                "Note: Update names and PRNs before submission.",
            ],
        ),
        (
            "Introduction",
            [
                "IT service desks receive thousands of support tickets daily.",
                "Manual triaging causes misrouting, slow resolution, and repeated fixes.",
                "Inspired by NASSCOM Hackathon use case: Intelligent Ticket Routing Agent.",
                "We built an end-to-end AI system that classifies, routes, retrieves resolutions, and escalates when uncertain.",
                "Tech stack: NLP, Sentence Transformers, RAG (ChromaDB), LangGraph, FastAPI.",
            ],
        ),
        (
            "Problem Statement",
            [
                "Misrouted tickets sent to wrong departments (Network vs Database, Security vs Access).",
                "Slow resolution due to manual categorization and triaging.",
                "Inconsistent classification across support agents.",
                "Past resolutions not reused — same issues solved from scratch.",
                "No confidence-aware mechanism to decide auto-resolve vs human escalation.",
            ],
        ),
        (
            "Objectives of the Project",
            [
                "Automatically classify IT tickets into 7 categories with confidence scores.",
                "Route tickets to the correct support department.",
                "Retrieve similar past tickets using RAG for resolution suggestions.",
                "Escalate to human agents when classifier or retrieval confidence is low.",
                "Evaluate multiple ML models (baselines, LogReg, SVM, SetFit).",
                "Build a working web portal for employees, agents, and admin analytics.",
            ],
        ),
        (
            "Project Plan with Timeline",
            [
                "Phase 1 (Done) — Dataset generation via Gemini API (~994 tickets, 7 categories)",
                "Phase 2 (Done) — Model training: TF-IDF+NB, Embeddings+LogReg/SVM, SetFit",
                "Phase 3 (Done) — RAG knowledge base (ChromaDB) + leakage fix (train-only index)",
                "Phase 4 (Done) — LangGraph agent: classify → route → retrieve → decide",
                "Phase 5 (Done) — FastAPI backend + Employee/Agent/Admin frontend",
                "Phase 6 (In Progress) — Scale dataset to 4k–5k, feedback loop, deployment",
            ],
        ),
        (
            "Literature Review",
            [
                "Ticket classification: TF-IDF + traditional ML vs transformer embeddings.",
                "Sentence-BERT / MiniLM for semantic text representation in short text tasks.",
                "SetFit: contrastive fine-tuning for few-shot text classification.",
                "RAG: retrieval-augmented generation for knowledge-grounded support answers.",
                "ITSM automation: routing rules, confidence thresholds, human-in-the-loop escalation.",
                "Agentic workflows: LangGraph for multi-step decision pipelines.",
            ],
        ),
        (
            "Gap in Research / Technology / Methodology",
            [
                "Most systems use keyword matching or single-model classification without retrieval.",
                "Few combine classification confidence + RAG similarity for escalation decisions.",
                "Evaluation leakage in RAG systems often inflates retrieval scores on test data.",
                "Small synthetic datasets limit fine-tuned models (SetFit overfits on val, not test).",
                "Lack of calibration-aware escalation in typical ITSM rule engines.",
            ],
        ),
        (
            "Description of the Proposed Solution",
            [
                "Hybrid AI pipeline: Classification + Routing + RAG + Agentic Decision.",
                "Step 1: Preprocess ticket (title + description) → embedding.",
                "Step 2: Classify category (LogReg or SetFit) → confidence score.",
                "Step 3: Map category → department (7 fixed mappings).",
                "Step 4: RAG retrieval from ChromaDB (695 train tickets only).",
                "Step 5: LangGraph agent decides auto-resolve OR escalate using dual thresholds.",
            ],
        ),
        (
            "Requirement Analysis",
            [
                "Functional: Submit ticket, view AI decision, agent queue, admin analytics.",
                "Functional: Choose classifier model (LogReg / SetFit) per submission.",
                "Functional: Show suggested resolution from similar past tickets.",
                "Non-functional: Response within ~2–5 sec after model warm-up.",
                "Non-functional: Reproducible train/val/test splits; no RAG data leakage.",
                "Data: 994 tickets, 7 balanced categories, augmented edge cases.",
            ],
        ),
        (
            "Technology Stack",
            [
                "Language: Python 3.x",
                "LLM / Data: Google Gemini API (synthetic ticket generation)",
                "ML: scikit-learn, Sentence-Transformers (MiniLM-L6-v2), SetFit",
                "Vector DB: ChromaDB via LangChain",
                "Agent: LangGraph state machine",
                "Backend: FastAPI + SQLite | Frontend: HTML/CSS/JS",
                "Version Control: Git + GitHub",
            ],
        ),
        (
            "System Design",
            [
                "Architecture: Employee Portal → FastAPI → LangGraph Agent → Response",
                "Agent graph: classify → route → retrieve → decide (auto-resolve / escalate)",
                "RAG index: built from train.csv only (695 docs) — prevents test leakage",
                "Dual thresholds: classifier confidence ≥ 0.50 AND RAG similarity ≥ 0.40",
                "Database: SQLite stores tickets, decisions, confidence, classifier model used.",
                "Portals: Employee (submit), Agent (escalated queue), Admin (analytics).",
            ],
        ),
        (
            "Development / Implementation",
            [
                "Data pipeline: generate → validate → split → augment edge cases.",
                "Trained 4 classifiers; deployed LogReg + SetFit in live API.",
                "Built Chroma vector store; fixed leakage (verify_rag_no_leakage.py).",
                "Implemented LangGraph agent in api/agent.py.",
                "FastAPI endpoints: submit-ticket, agent queue, analytics, classifiers.",
                "Frontend: login, employee model picker, agent dashboard, admin metrics.",
            ],
        ),
        (
            "Testing & Debugging",
            [
                "Unit smoke test: test_single_ticket.py (both LogReg and SetFit).",
                "Model evaluation: compare_models.py, 5-fold CV, held-out eval set.",
                "Calibration study: reliability diagram — model is underconfident.",
                "Error analysis: confusion matrix; errors on adjacent categories.",
                "RAG verification: no exact memorization of test tickets (similarity ~0.22).",
                "Integration: live API tested on http://127.0.0.1:8000/app/",
            ],
        ),
        (
            "Project Outcome",
            [
                "Working end-to-end prototype with web UI and dual classifier support.",
                "Test accuracy: TF-IDF+NB 65.3% | LogReg 70.0% | SVM 70.0% | SetFit 70.7%",
                "5-fold CV: 73.4% ± 0.96% accuracy (stable across folds).",
                "Held-out eval: 67.2% accuracy (good generalization).",
                "Confidence calibration supports safe escalation strategy.",
                "RAG leakage identified and fixed — reproducible verification script.",
            ],
        ),
        (
            "Conclusion",
            [
                "Successfully built an AI ticket routing agent combining NLP, RAG, and agentic logic.",
                "Embedding-based classifiers outperform TF-IDF baseline by ~4.7 F1 points.",
                "Dual-threshold escalation reduces risk of wrong auto-resolution.",
                "SetFit shows promise but needs larger dataset (4k–5k scale-up planned).",
                "Future: feedback loop, conformal prediction, production deployment.",
            ],
        ),
        (
            "References",
            [
                "1. Reimers & Gurevych — Sentence-BERT (2019)",
                "2. Tunstall et al. — SetFit: Efficient Few-Shot Learning (2022)",
                "3. Lewis et al. — Retrieval-Augmented Generation (2020)",
                "4. NASSCOM — AI Powered Intelligent Ticket Routing Use Case",
                "5. scikit-learn, HuggingFace Transformers, LangChain, LangGraph documentation",
                "6. Kaggle IT Support Ticket datasets (reference validation)",
            ],
        ),
        (
            "Presentation Photograph",
            [
                "[Insert photograph of the presentation here]",
                "",
                "To be added after due permission from the guide.",
                "",
                "Date: _______________",
                "Venue: Department of AIML — Second Presentation (CA-3)",
            ],
        ),
    ]

    for title, lines in slides_data:
        add_content_slide(prs, title, lines)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    prs.save(OUTPUT)
    try:
        prs.save(OUTPUT_COPY)
        print(f"Saved copy: {OUTPUT_COPY}")
    except Exception as exc:
        print(f"Could not save to Downloads: {exc}")
    print(f"Saved: {OUTPUT}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
