# Stage 2 Hidden Test Evaluation

## Result

The exact Stage 1 checkpoint was loaded from:

```text
model_checkpoint/sentiment_ensemble.joblib
```

No retraining, fine-tuning, refitting, or threshold modification was performed.

**Checkpoint SHA-256**

```text
c05617833b4b2f2a5dbc2f660350644691c2ac3d58038813e1eaf9eee91db23e
```

## Hidden Test Performance

- Hidden test examples: **600**
- Negative examples: **300**
- Positive examples: **300**
- Accuracy: **0.7700 (77.0%)**
- Balanced accuracy: **0.7700**

### Confusion Matrix

Rows are true labels and columns are predicted labels.

```text
[[225, 75],
 [63, 237]]
```

- True negatives: **225**
- False positives: **75**
- False negatives: **63**
- True positives: **237**

## Public vs. Hidden Test

The Stage 1 public-test accuracy was **77.0%**.  
The Stage 2 hidden-test accuracy is also **77.0%**.

The identical total accuracy suggests that the Stage 1 model generalized to the unseen hidden set at about the same level as it did on the public set. The error distribution changed somewhat, but there was no overall accuracy drop.

## If I Had More Time or Compute

I would compare this sparse TF-IDF ensemble with a compact pretrained language model or pretrained sentence embeddings. With only 240 labeled training examples, I would avoid training a large neural network from scratch. I would also use repeated stratified cross-validation on the training set to make model selection less sensitive to a single fold split.

## Use of AI

Generative AI was used to assist with Stage 2 code organization, checkpoint verification, evaluation formatting, and generation of the required prediction file. The Stage 2 model itself was not retrained or modified.
