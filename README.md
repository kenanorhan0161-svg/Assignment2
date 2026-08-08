# Stage 1 Sentiment Classification

## What this repository does

This project predicts the sentiment of movie reviews:

- `0` → negative
- `1` → positive

The Stage 1 model is trained only on `train.csv`.  
`public_test.csv` is reserved for evaluation and prediction generation.

The final classifier is intentionally lightweight because the released training set contains only 240 reviews and is strongly imbalanced.

---

## Repository layout

```text
.
├── stage1_notebook.ipynb
├── README.md
├── requirements.txt
├── predict.py
├── public_test_predictions.csv
└── model_checkpoint/
    ├── sentiment_ensemble.joblib
    └── config.json
```

### Important files

**`stage1_notebook.ipynb`**  
Contains the complete Stage 1 workflow: inspection, model design, cross-validation, threshold calibration, final fitting, public-test evaluation, checkpoint saving, and reload verification.

**`model_checkpoint/sentiment_ensemble.joblib`**  
Stores the fitted word TF-IDF model, character TF-IDF model, ensemble weight, and classification threshold.

**`model_checkpoint/config.json`**  
Stores readable metadata about the submitted checkpoint.

**`public_test_predictions.csv`**  
Contains exactly:

```text
id,predicted_label
```

---

## Final model idea

Instead of training a large neural network on a very small dataset, I used two regularized sparse-text models and combined their decision scores.

### Branch A — word features

- TF-IDF word features
- unigrams + bigrams
- sublinear term-frequency scaling
- Linear SVM
- balanced class weights

### Branch B — character features

- TF-IDF character features
- character 3-grams through 5-grams
- Linear SVM
- balanced class weights

### Ensemble

The final score is:

```text
ensemble_score
    = 0.85 * word_model_score
    + 0.15 * character_model_score
```

The prediction rule is:

```text
predict 1 if ensemble_score >= 0.267951
otherwise predict 0
```

The ensemble weight and decision threshold were selected from **out-of-fold predictions produced from `train.csv` only**.

---

## Why this approach fits the assignment

The main difficulty is not raw model capacity. It is getting useful generalization from a training set with:

```text
180 positive
60 negative
```

That creates three problems:

1. a classifier can become biased toward positive reviews,
2. a high-capacity model can memorize a tiny dataset,
3. exact tokens appearing in evaluation data may not appear in training.

The pipeline addresses them as follows.

### Class imbalance

Both SVMs use:

```python
class_weight="balanced"
```

I also calibrated the final ensemble threshold using balanced accuracy rather than assuming the default SVM cutoff was ideal.

### Small sample size

A sparse linear model has far fewer opportunities to overfit than training a recurrent network or Transformer from scratch.

Regularization is controlled through the SVM `C` parameter.

### Unseen vocabulary

The character branch does not depend entirely on whole-word matches. Character n-grams can still respond to morphology, punctuation patterns, word fragments, and related surface forms even when an exact word was absent from the training vocabulary.

---

## Training choices

| Setting | Value |
|---|---|
| Cross-validation | 5-fold StratifiedKFold |
| Random seed | 42 |
| Word SVM `C` | 0.5 |
| Character SVM `C` | 0.25 |
| Word n-grams | 1–2 |
| Character n-grams | 3–5 |
| Word feature cap | 50,000 |
| Character feature cap | 60,000 |
| Class weighting | balanced |
| Threshold metric | balanced accuracy |

There is no neural-network batch size or learning rate because the final classifier uses `LinearSVC` rather than SGD/Adam training.

---

## Public test performance

The submitted checkpoint produced:

**Accuracy:** `0.7700` (77.0%)

**Balanced accuracy:** `0.7700`

**Confusion matrix:**

```text
[[163, 37],
 [55, 145]]
```

Rows are true labels and columns are predicted labels.

That corresponds to:

- 163 correctly classified negative reviews
- 37 negative reviews predicted as positive
- 55 positive reviews predicted as negative
- 145 correctly classified positive reviews

---

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Place these two files in the repository root:

```text
train.csv
public_test.csv
```

Then launch:

```bash
jupyter notebook stage1_notebook.ipynb
```

Run the notebook from top to bottom.

---

## Inference without retraining

The submitted checkpoint can be loaded directly with:

```bash
python predict.py public_test.csv reproduced_predictions.csv
```

For Stage 2, the same saved checkpoint can be used on the hidden file:

```bash
python predict.py hidden_test.csv hidden_test_predictions.csv
```

No fitting is required for either command.

---

## Use of AI

Generative AI was used as a development assistant during Stage 1.

AI assistance included:

- discussing model families that make sense for a very small text dataset;
- comparing a simpler TF-IDF/linear-classifier approach against heavier neural approaches;
- suggesting ways to handle the 3:1 class imbalance;
- helping organize stratified cross-validation and train-only threshold calibration;
- helping structure checkpoint-saving and reload verification code;
- helping format the notebook and repository documentation;
- helping check that the prediction CSV has the exact required column names.

The AI-related requests in this work focused on practical implementation questions such as building a laptop-friendly sentiment classifier, preventing the majority class from dominating predictions, preserving a reloadable Stage 1 checkpoint, and presenting the evaluation clearly.

The final submitted workflow uses only the released training examples for fitting. The public test set is not included in the training process.
