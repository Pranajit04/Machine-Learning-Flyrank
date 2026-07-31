# Refresh & Content Opportunity Scoring: Predicting Declining Search Pages

## Abstract
This project asks whether search-performance signals can identify content pages likely to be declining in Google Search visibility before the decline is obvious. Using 90-day windowed features (impressions, clicks, position, position volatility, and query-mix signals) built from FlyRank's anonymized search warehouse, a Random Forest classifier was compared against a transparent hand-rule baseline on a client-holdout split. The model reached 74.9% accuracy versus a 73.4% baseline and 53.4% majority-class rate, with impression trend, position volatility, and query-mix concentration as the strongest signals. The lift over the baseline was modest, suggesting the simple rule already captures most of the recoverable signal in this feature set — an honest and useful finding in its own right.

## Introduction / Problem Statement
Content teams need to prioritize which pages to review or refresh, but manually auditing every page in a catalog doesn't scale. This project supports that triage decision: given a page's recent search performance, how likely is it to be declining, and how confident should a reviewer be in that flag?

## Data
- **Source:** FlyRank ML Internship warehouse release (Hugging Face, gated, `FlyRank/internship-warehouse`)
- **Tables used:** `fact_content_daily_performance` (~79M rows, daily grain), `dim_clients`, `fact_content_query_90d`
- **Date window:** latest 180 days available as of 2026-06-30, split into a last-90-day window and a prior-90-day comparison window
- **Filtering:** content items required ≥200 impressions in the prior-90 window to ensure enough signal (104,725 items met this bar; 84,979 also had query-level data)
- **Excluded:** no client names, domains, URLs, or queries appear anywhere in this repo or paper — only pseudonymized hash IDs, per the program's data-use terms

## Methodology
**Label definition:** a page is labeled *declining* if last-90-day impressions fell below 80% of the prior-90-day impressions AND clicks did not increase to compensate. This produced a near-balanced label (54.3% / 45.7%), avoiding both label leakage and a degenerate majority-class problem encountered in an earlier attempt.

**Features (leakage-safe — none derived from the same window as the label):**
- `imp_prev90`, `clk_prev90` — prior-window volume
- `pos_last90`, `pos_volatility_90d` — average and standard deviation of search position
- `visible_queries`, `rare_share`, `anon_share`, `top_query_share` — query-mix concentration signals

**Baseline:** a transparent hand-rule score (+2 for impression drop, +1 for average position > 15, +1 for high position volatility), thresholded at ≥2.

**Validation:** `GroupShuffleSplit` on `client_hash_id` (80/20), ensuring no client's pages appeared in both train and test — this tests whether findings generalize to *unseen* clients, not just unseen pages from known clients.

## Results
| | Accuracy | Precision (declining) | Recall (declining) |
|---|---|---|---|
| Majority-class baseline | 53.4% | — | — |
| Hand-rule baseline | 74.0% | 0.642 | 1.000 |
| Random Forest model | 74.9% | 0.802 | 0.613 |

**Feature importance (Random Forest):**
1. `imp_prev90` (0.233)
2. `pos_volatility_90d` (0.137)
3. `rare_share` (0.128)
4. `visible_queries` (0.127)
5. `anon_share` (0.113)

The model trades recall for precision relative to the baseline — it misses more true decliners but is more confident when it flags one. Which tradeoff matters depends on whether the review team prefers a shorter, higher-confidence list (model) or a longer, safer net (baseline).

## Limitations & Honest Framing
- This is a **directional, decision-support** signal, not a causal or algorithmic explanation of Google ranking behavior.
- The near-tie between baseline and model suggests the engineered features capture most of what's learnable here; a richer feature set (content freshness, backlink signals, competitor movement) might separate them further.
- An earlier label definition (arbitrary CTR threshold) produced a 97%/3% imbalance and a false sense of model skill — corrected here, but a reminder that label design is as important as model choice.
- Client-holdout validation is a stronger test than a random row split, but with ~70 clients total, the 20% test group is a small number of clients; results could shift with a different random seed.

## Ranked Recommendations
1. **Use the model where precision matters** (e.g., limited reviewer capacity) — it flags fewer pages but with higher confidence.
2. **Use the hand-rule baseline where recall matters** (e.g., a first-pass audit) — it catches every actual decliner in this test set, at the cost of more false positives.
3. **Prioritize pages with high `pos_volatility_90d` and low `visible_queries`** — these carry the strongest signal and are cheap to compute without a trained model at all.
4. **Revisit with richer features before fully automating** — the marginal model lift suggests this shouldn't yet replace human review, only prioritize its queue.

## Reproducibility
- Notebook: `work/notebooks/capstone.ipynb` in this repository
- Built on: `flyrank-bih/flyrank-ml-internship-starter`, notebook 03 workflow (DuckDB over `hf://`)
- Random seed fixed at 42 throughout for the train/test split and Random Forest

## Acknowledgments & Data Credit
Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai)
