# Spark MLlib Fake News Detection Pipeline

A scalable fake news detection pipeline built with **Spark MLlib**, comparing Logistic Regression, Random Forest, Gradient Boosted Trees, and a stacked ensemble on the LIAR dataset — benchmarked across a single-node Colab environment and a distributed **GCP Dataproc** cluster.

## Problem & Motivation

Fake news detection is a classification problem: given a statement, predict whether it's real or fake. Manual fact-checking doesn't scale — social media platforms generate millions of statements daily, and misinformation poses a direct threat to democratic integrity and public health.

This is a strong candidate for a distributed ML approach because:
- The classification signal is subtle — fake news often mimics the language patterns of real news, so simple rule-based approaches fail.
- Text data is unstructured and high-dimensional once vectorized (TF-IDF), requiring efficient distributed processing.
- Spark MLlib scales horizontally across a cluster, making the pipeline applicable to real-world content moderation systems operating at scale.

## Dataset

**[LIAR dataset](https://doi.org/10.18653/v1/P17-2067)** (Wang, 2017) — 12,836 short political statements labelled across 6 veracity classes: `true`, `mostly-true`, `half-true`, `barely-true`, `false`, `pants-fire`.

The 6 classes were collapsed into a **binary label**:
- **Real (0):** true, mostly-true, half-true
- **Fake (1):** barely-true, false, pants-fire

This simplifies the classification task, produces a more balanced class split (56% / 44%), and reflects the practical binary decision fake news detection ultimately requires.

## Pipeline

All models share the same feature engineering pipeline, ensuring the classifier choice is the only variable across comparisons:

| Stage | Transformer/Estimator | Purpose |
|---|---|---|
| 1 | `Tokenizer` | Splits raw statement text into words |
| 2 | `StopWordsRemover` | Removes uninformative common words |
| 3 | `HashingTF` | Converts words into a 10,000-feature term frequency vector |
| 4 | `IDF` | Downweights terms common across all documents |
| 5 | `FeatureHasher` | Hashes high-cardinality speaker/party metadata into a 1,000-feature vector |
| 6 | `VectorAssembler` | Combines text + metadata features into a single vector |
| 7 | Classifier | Logistic Regression / Random Forest / GBT / Stacking |

`FeatureHasher` was used instead of `OneHotEncoder` for categorical features (speaker, party) to avoid the curse of dimensionality — with thousands of unique speakers, one-hot encoding would explode the feature space and increase sparsity without improving performance.

## Exploratory Analysis

Speaker credibility count features (historical true/false counts) were checked for correlation with the binary label before modelling. All five showed very weak correlation (strongest: `pants_fire` at 0.16), suggesting a speaker's historical credibility alone is a poor predictor of any single statement's veracity — reinforcing that the **text content itself** carries most of the predictive signal.

## Models & Results

| Model | AUC | Accuracy | F1 | Train Time |
|---|---|---|---|---|
| Logistic Regression | 0.56 | 0.55 | 0.55 | 2.58 s |
| Random Forest | 0.74 | 0.59 | 0.47 | 13.55 s |
| **GBT** | **0.81** | **0.73** | **0.73** | 53.79 s |
| Stacking (LR + RF) | 0.74 | 0.68 | 0.68 | — |

**Metrics used:** AUC (primary — threshold-independent, handles class imbalance), Accuracy (interpretable, appropriate given the roughly balanced 56/44 split), and F1 (balances false positives and false negatives, both of which carry real-world cost in a content moderation setting).

**Key findings:**
- Logistic Regression performed barely better than random guessing (AUC 0.56), consistent with the expectation that a linear decision boundary can't capture the non-linear, interaction-heavy relationship between text patterns and speaker context (e.g. the same phrase can be truthful from one speaker and misleading from another).
- GBT significantly outperformed all other models (AUC 0.81, accuracy 73.3%), exceeding the typical 65–70% state-of-the-art range reported for the LIAR dataset — likely due to combining TF-IDF text features with speaker credibility metadata.
- GBT's near-identical precision and recall (both 0.73) indicate a well-balanced model, avoiding both over-flagging real news and missing fake statements.
- GBT did not converge in a reasonable time on a single-node Colab environment, underscoring the need for distributed infrastructure for computationally expensive sequential ensemble methods.

## Distributed Computing: Single Node vs. Cluster

| Environment | Nodes | LR Training Time |
|---|---|---|
| Google Colab | 1 | 5.97 s |
| GCP Dataproc | 3 (1 master, 2 workers) | 2.58 s |

Training on the GCP Dataproc cluster was **~2.3x faster** than on a single node for Logistic Regression. The scalability tradeoff across models:
- **Logistic Regression** — fastest and most scalable, minimal inter-node communication, but underfits complex patterns.
- **Random Forest** — moderately expensive but highly parallelizable, since each tree trains independently.
- **GBT** — most expensive and least parallelizable, since boosting rounds are inherently sequential (each tree depends on the previous one's errors) — but achieves the best predictive performance.

At the scale of millions of real-world social media posts, a single machine would become a hard bottleneck, making distributed infrastructure a necessity rather than a convenience.

## Limitations

- **Dataset size** — 12k rows is relatively small for a big-data pipeline; scalability benefits are modest at this size and would be far more pronounced at production scale.
- **Label collapsing** — reducing 6 nuanced veracity classes to a binary label introduces noise, as ambiguous statements are forced into discrete categories.
- **Short, sparse text** — statements average ~17 words, producing very sparse TF-IDF vectors (most of the 10,000 features are zero for any given statement).

## Tech Stack

- **Spark MLlib** (PySpark)
- **GCP Dataproc** (distributed cluster: 1 master, 2 workers)
- **Google Colab** (single-node baseline comparison)

## Future Work

With richer data — full article text, source metadata, and social network propagation patterns — this pipeline could be extended into a production-ready content moderation system. The architecture itself is general enough to adapt to other misinformation detection problems beyond political statements.

## References

- Wang, W. Y. (2017). "Liar, liar pants on fire": A new benchmark dataset for fake news detection. *ACL 2017*, 422–426.
- Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009). Feature hashing for large scale multitask learning. *ICML 2009*, 1113–1120.
- Huang, J.-Y., & Liu, J.-H. (2020). Using social media mining technology to improve stock price forecast accuracy. *Journal of Forecasting*, 39(1), 104–116.
- Sivakumar, M., Parthasarathy, S., & Padmapriya, T. (2024). Trade-off between training and testing ratio in machine learning for medical image processing. *PeerJ Computer Science*, 10, e2245.
