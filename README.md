# Task_04_Descriptive_Stats

This repository contains code and analysis for **Research Task 4**, which explores descriptive statistics using three approaches:

* ✅ **Pure Python** (standard library only)
* 🐼 **Pandas** (popular data analysis library)
* 🦓 **Polars** (high-performance DataFrame library)

The goal is to produce consistent descriptive statistics and visualizations on social media datasets related to the **2024 US Presidential election**.

---

## 📦 Datasets

Three real-world datasets analyzed (files **not included in the repo**):

| Dataset name   | File name                                 |    Rows |
| -------------- | ----------------------------------------- | ------: |
| Facebook Ads   | `2024_fb_ads_president_scored_anon.csv`   | 246,745 |
| Facebook Posts | `2024_fb_posts_president_scored_anon.csv` |  19,009 |
| Twitter Posts  | `2024_tw_posts_president_scored_anon.csv` |  27,304 |

---

## 📂 Repository Contents

| File                                                      | Description                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| `pure_python_stats.py`                                    | Descriptive statistics using only the Python standard library |
| `pandas_stats.py`                                         | Analysis & visualizations using Pandas                        |
| `polars_stats.py`                                         | Analysis & visualizations using Polars                        |

---

## 🛠 How to Run

1️⃣ Install dependencies:

```bash
pip install pandas polars matplotlib seaborn
```

2️⃣ Download the datasets (do **not** include them in GitHub).

3️⃣ Run the analysis scripts:

```bash
python pure_python_stats.py
python pandas_stats.py
python polars_stats.py
```

---

## 🔑 Key Insights from the Analysis

### 📦 **Facebook Ads**

* Ad budgets are heavily skewed: most ads have low spend, but a few spend tens of thousands.
* Unequal reach: some ads reach millions, revealing big exposure gaps.
* Strong targeting: `delivery_by_region` and `demographic_distribution` fields show clear geo-demographic strategies.
* Frequent topics: health, governance, immigration, and election integrity dominate.
* Few dominant pages: a small number of pages publish most ads.
* Reused ads: shared ad\_ids suggest strategies like A/B testing.

---

### 📄 **Facebook Posts**

* Engagement is highly skewed: few posts get most likes, shares, and comments.
* Emotional reactions like Love, Haha, Angry provide insight into tone.
* Sponsored posts generally outperform unsponsored posts (via Overperforming Score).
* Longer videos tend to have higher view counts.
* Focus topics: governance, education, immigration.
* Admin location: majority US-based; some foreign-administered pages could raise influence concerns.

---

### 🐦 **Twitter Posts**

* Retweets dominate content over original tweets.
* Engagement is highly skewed: few tweets get most likes and replies.
* Topics: race, governance, foreign policy appear often.
* Controlled conversations: `isConversationControlled` indicates attempts to limit replies.
* CreatedAt & month-year fields help detect spikes during major events.

---

## 📊 Findings & Comparison of Approaches

| Approach    | Speed (approx.) | Ease of use |                   Suitability |
| ----------- | --------------: | ----------: | ----------------------------: |
| Polars      |         \~1 sec |    Moderate | Best for large/fast pipelines |
| Pandas      |       \~1–2 sec |        Easy |         Balanced; widely used |
| Pure Python |          Slower |      Harder |     Educational; full control |

✅ All approaches produced consistent results (counts, means, min, max, std dev).
⚡ Polars was fastest thanks to its Rust engine and parallelism.
🐼 Pandas offered best trade-off between speed and readability.
🐍 Pure Python required most manual coding, but gives transparency.

