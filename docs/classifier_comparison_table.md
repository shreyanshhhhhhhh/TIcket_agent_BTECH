# LogReg vs SetFit — Full Classifier Comparison

Dataset: 695 train / 149 val / 150 test tickets · 7 categories · Backbone: `all-MiniLM-L6-v2`

---

## 1. Model Architecture & Approach

| Aspect | Logistic Regression (LogReg) | SetFit |
|--------|------------------------------|--------|
| **Full name** | Embeddings (MiniLM) + Logistic Regression | SetFit contrastive fine-tuned MiniLM |
| **Embedding model** | all-MiniLM-L6-v2 (frozen) | all-MiniLM-L6-v2 (fine-tuned) |
| **Classifier head** | sklearn LogisticRegression | SetFit classification head |
| **Training method** | Encode once → train linear classifier on fixed embeddings | Contrastive fine-tuning + classification (20 iterations, 1 epoch) |
| **Class balancing** | `class_weight="balanced"` | Handled by SetFit Trainer |
| **Fine-tunes transformer?** | No | Yes |
| **Model artifact** | `logreg_classifier.joblib` | `models/setfit_classifier/` folder |
| **Extra dependencies** | scikit-learn, joblib | setfit, datasets, PyTorch, transformers |

---

## 2. Performance — Validation & Test (same splits)

| Metric | LogReg | SetFit | Better |
|--------|--------|--------|--------|
| **Validation Accuracy** | 73.15% | **79.87%** | SetFit (+6.7 pts) |
| **Validation F1 (macro)** | 73.43% | **80.10%** | SetFit (+6.7 pts) |
| **Test Accuracy** | 70.00% | **70.67%** | SetFit (+0.7 pts) |
| **Test F1 (macro)** | 70.26% | 70.63% | SetFit (+0.4 pts) |
| **Test F1 (weighted)** | 69.87% | **70.49%** | SetFit |
| **Test Precision (macro)** | 71.19% | **71.66%** | SetFit |
| **Test Recall (macro)** | 70.43% | **70.69%** | SetFit |
| **Val → Test drop (overfit gap)** | 3.15 pts | **9.20 pts** | LogReg (more stable) |
| **Misclassified (test / 150)** | 45 | **44** | SetFit (1 fewer error) |

---

## 3. Performance — Held-Out Eval (119 tickets, never used in training)

| Metric | LogReg | SetFit | Better |
|--------|--------|--------|--------|
| **Holdout Accuracy** | **67.23%** | 62.18% | LogReg (+5.0 pts) |
| **Holdout F1 (macro)** | **66.99%** | 62.35% | LogReg (+4.6 pts) |

---

## 4. Per-Category F1 (Test Set)

| Category | LogReg | SetFit | Better |
|----------|--------|--------|--------|
| Access Management | 63.4% | **70.0%** | SetFit |
| Application | **75.7%** | 72.3% | LogReg |
| Database | **92.7%** | 82.9% | LogReg |
| Infrastructure | 48.8% | **57.8%** | SetFit |
| Network | 62.7% | **68.1%** | SetFit |
| Security | 72.3% | **76.6%** | SetFit |
| Storage | **76.2%** | 66.7% | LogReg |
| **Categories SetFit wins** | 2 | **5** | — |

---

## 5. Confidence & Calibration (Test Set)

| Aspect | LogReg | SetFit |
|--------|--------|--------|
| **Avg confidence (all predictions)** | 0.528 | 0.861 |
| **Avg confidence (correct)** | 0.560 | 0.904 |
| **Avg confidence (wrong)** | 0.454 | 0.759 |
| **Min / Max confidence** | 0.24 / 0.86 | 0.40 / 0.96 |
| **Calibration behavior** | **Underconfident** — actual accuracy often higher than stated confidence | **Overconfident** — high scores even on wrong predictions (0.76 avg when wrong) |
| **Escalation impact** | More tickets escalate (confidence often below 0.50 threshold) | More tickets pass confidence threshold; relies more on RAG similarity gate |
| **Reliability diagram** | Available (`docs/reliability_diagram.png`) | Not yet generated |

---

## 6. Operational & Deployment

| Aspect | LogReg | SetFit |
|--------|--------|--------|
| **Model size on disk** | **~0.01 MB** | ~87.36 MB |
| **Inference latency (per ticket)** | ~12.6 ms | ~12.5 ms |
| **Startup load time** | Fast (joblib + shared embedder) | Slower (loads full transformer + head) |
| **RAM at runtime** | Lower (shared MiniLM embedder only) | Higher (second full model in memory) |
| **Training time (CPU)** | **~2–5 minutes** | ~60–75 minutes |
| **GPU required?** | No | No (but faster with GPU) |
| **Git-friendly** | Yes (tiny joblib file) | Large (86.65 MB safetensors; GitHub warns >50 MB) |
| **Retrain complexity** | Low — re-encode + refit | High — full SetFit training pipeline |
| **API support** | Yes (`classifier_model=logreg`) | Yes (`classifier_model=setfit`) |
| **Default in UI** | Yes (baseline) | Optional (employee picker) |

---

## 7. Cross-Validation & Stability

| Aspect | LogReg | SetFit |
|--------|--------|--------|
| **5-fold CV accuracy** | **73.44% ± 0.96%** | Not run |
| **5-fold CV F1 (macro)** | **73.44% ± 1.10%** | Not run |
| **Prediction agreement (test set)** | — | 73.3% of predictions match LogReg |
| **Generalization (holdout vs test)** | −2.8 pts (70.0 → 67.2) | −8.5 pts (70.7 → 62.2) |

---

## 8. Strengths & Weaknesses

| | LogReg | SetFit |
|---|--------|--------|
| **Strengths** | Fast to train; tiny model; stable val→test; best holdout generalization; underconfidence safer for escalation; easy to retrain | Best validation scores; wins on 5/7 categories on test; contrastive learning captures semantic nuance; state-of-the-art few-shot method |
| **Weaknesses** | Weakest on Infrastructure (48.8% F1); frozen embeddings limit adaptation | Large model; long training; severe val→test gap (overfitting on small data); worse holdout performance; overconfident scores; not Git-friendly |

---

## 9. Recommendation for This Project

| Use case | Recommended |
|----------|-------------|
| **Production default (current data size)** | **LogReg** — better holdout generalization, safer confidence, tiny footprint |
| **Research / ablation experiments** | **SetFit** — demonstrates contrastive fine-tuning; needs 4k–5k dataset scale-up |
| **Escalation-sensitive deployment** | **LogReg** — underconfident scores reduce false auto-resolves |
| **After dataset scale-up (planned)** | Re-evaluate SetFit — expected to close val/test gap |

---

*Generated from `models/logreg_results.json`, `models/setfit_results.json`, test predictions, and live benchmark scripts.*
