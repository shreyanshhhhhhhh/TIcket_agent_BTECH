# Phase 2: Model Development & Evaluation

## Overview

Phase 2 focuses on developing, training, and evaluating multiple machine learning models for automated IT support ticket classification. The objective was to compare a traditional text classification approach with semantic embedding-based models and identify the best-performing classifier for deployment in the intelligent ticket routing system.

The evaluation process included model comparison, cross-validation, held-out testing, confidence calibration, and error analysis to ensure that the selected model performs reliably on unseen data.

---

# Objectives

The primary goals of Phase 2 were:

- Develop a baseline machine learning classifier.
- Train embedding-based classifiers using sentence transformers.
- Compare different classification algorithms.
- Measure model stability using cross-validation.
- Evaluate real-world generalization using a held-out dataset.
- Analyze prediction confidence and common classification errors.

---

# Model Development

Three classification models were implemented.

| Model | Feature Representation | Classifier |
|--------|------------------------|------------|
| Baseline | TF-IDF | Multinomial Naive Bayes |
| Main Model | Sentence Embeddings (all-MiniLM-L6-v2) | Logistic Regression |
| Comparison Model | Sentence Embeddings (all-MiniLM-L6-v2) | Support Vector Machine |

The sentence embedding model converts each ticket into a dense semantic vector before classification, allowing the classifier to understand contextual meaning instead of relying solely on keywords.

---

# Experimental Setup

The dataset was divided into:

- Training Set
- Validation Set
- Test Set
- Independent Held-Out Evaluation Set

Performance was measured using:

- Accuracy
- Macro F1 Score
- Precision
- Recall

Additional evaluation techniques included:

- 5-Fold Cross Validation
- Confusion Matrix Analysis
- Confidence Calibration
- Error Analysis

---

# Model Performance Comparison

| Model | Validation Accuracy | Validation F1 | Test Accuracy | Test F1 |
|--------|-------------------:|--------------:|--------------:|---------:|
| TF-IDF + Naive Bayes | 66.44% | 67.19% | 65.33% | 65.64% |
| Embeddings + Logistic Regression | **73.15%** | **73.43%** | **70.00%** | **70.26%** |
| Embeddings + SVM | **73.15%** | **73.36%** | **70.00%** | **69.97%** |

## Observation

Sentence embeddings improved the Macro F1 score by approximately **4.7 percentage points** compared with the traditional TF-IDF baseline.

Both embedding-based classifiers achieved nearly identical performance, with Logistic Regression slightly outperforming SVM in terms of Macro F1 score.

---

# Cross Validation

To verify model stability, five-fold cross validation was performed using the complete training dataset.

| Metric | Mean | Standard Deviation |
|---------|-----:|-------------------:|
| Accuracy | **73.44%** | ±0.96% |
| Macro F1 | **73.44%** | ±1.10% |

## Interpretation

The low standard deviation indicates that the model performs consistently across multiple train-test splits.

The cross-validation accuracy is slightly higher than the single test accuracy, suggesting that the selected test split was marginally more difficult rather than indicating overfitting.

---

# Held-Out Evaluation

A completely independent evaluation dataset was reserved throughout development and was never used during model training or hyperparameter tuning.

| Metric | Score |
|---------|------:|
| Accuracy | **67.23%** |
| Macro F1 | **66.99%** |

## Interpretation

Compared with the standard test accuracy of **70.00%**, the held-out evaluation shows only a small decrease of approximately **3 percentage points**.

This behaviour is expected for unseen real-world data and demonstrates that the model generalizes well instead of memorizing the training set.

---

# Confidence Calibration

Model confidence scores were compared against actual prediction accuracy.

| Confidence Range | Average Confidence | Actual Accuracy |
|-----------------|-------------------:|----------------:|
| <50% | 39% | 54% |
| 50–60% | 54% | 81% |
| 60–70% | 65% | 80% |
| 70–80% | 74% | 92% |
| 80–90% | 84% | 50%* |

*Only two predictions belonged to the highest confidence interval.

## Key Observation

Unlike many modern machine learning models that tend to be overconfident, the proposed classifier is systematically **underconfident**.

Across nearly every confidence interval, the actual prediction accuracy exceeded the reported confidence score.

This suggests that future probability calibration techniques such as Platt Scaling or Temperature Scaling could improve confidence estimation and enable lower automatic-routing thresholds without sacrificing prediction reliability.

---

# Error Analysis

Misclassified tickets were manually inspected to understand model behaviour.

Representative examples include:

| Actual Category | Predicted Category | Interpretation |
|----------------|-------------------|----------------|
| Security | Infrastructure | Printer-related security issue contains infrastructure terminology |
| Storage | Security | BitLocker encryption is strongly associated with security concepts |
| Infrastructure | Storage | RAID arrays naturally overlap with storage technologies |

These errors are semantically reasonable rather than random, indicating that the classifier has learned meaningful contextual relationships between technical domains.

---

# Discussion

The experimental results demonstrate several important findings.

- Sentence embeddings consistently outperform TF-IDF representations.
- Logistic Regression and SVM achieve comparable performance.
- Five-fold cross validation confirms stable model behaviour.
- Held-out evaluation demonstrates strong generalization capability.
- Confidence analysis indicates conservative probability estimates.
- Classification mistakes mainly occur between technically related categories rather than unrelated classes.

Overall, the embedding-based Logistic Regression model was selected as the final classifier due to its strong performance, lower computational complexity, and slightly higher Macro F1 score.

---

# Conclusion

Phase 2 successfully developed and evaluated multiple machine learning models for IT support ticket classification.

Compared with the TF-IDF baseline, semantic sentence embeddings significantly improved classification performance while maintaining stable results across validation, testing, and held-out evaluation datasets.

Comprehensive evaluation through cross-validation, calibration analysis, and manual error inspection confirms that the selected model is reliable for practical deployment in an intelligent ticket routing system.

The outcomes of this phase provide the foundation for **Phase 3**, where the trained classifier will be integrated into an automated routing pipeline capable of confidence-aware ticket assignment and escalation.

---

# Deliverables

✔ Baseline TF-IDF + Naive Bayes model

✔ Sentence Embeddings + Logistic Regression model

✔ Sentence Embeddings + Support Vector Machine model

✔ Model comparison report

✔ 5-Fold Cross Validation

✔ Held-Out Evaluation

✔ Confidence Calibration Analysis

✔ Error Analysis

✔ Confusion Matrix

✔ Final Model Selection