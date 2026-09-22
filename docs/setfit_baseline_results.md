# Classifier Comparison (695 train samples)

Generated from `notebooks/compare_models.py` on current train/val/test splits.

| Model | Val Accuracy | Val F1 (macro) | Test Accuracy | Test F1 (macro) |
|-------|-------------|----------------|---------------|-----------------|
| Baseline (TF-IDF + NB) | 66.4% | 67.2% | 65.3% | 65.6% |
| Main (Embeddings + LogReg) | 73.2% | 73.4% | 70.0% | 70.3% |
| SVM (Embeddings + SVM) | 73.2% | 73.4% | 70.0% | 70.0% |
| SetFit (Fine-tuned MiniLM) | 79.9% | 80.1% | 70.7% | 70.6% |

## Notes for paper

- SetFit improves **validation** accuracy (~+7 pts vs LogReg) but **test** accuracy is similar (~71%), suggesting limited generalization on the current ~994-ticket corpus.
- Scaling to 4k–5k tickets (see README dataset v2 workflow) is the planned ablation to test whether SetFit's contrastive training benefits from more data.
- Live API supports both LogReg and SetFit; employees choose per ticket. Admin analytics reports volume by `classifier_model`.
- RAG index is built from train split only (695 tickets) to prevent evaluation leakage.

## Reproduce

```powershell
python notebooks/compare_models.py
python notebooks/evaluate_setfit_holdout.py
python test_single_ticket.py
```
