# CorpusDiff

> **Transformer-based language corpus evaluation for dummies**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Made at Kiel University](https://img.shields.io/badge/Made%20at%20Kiel%20University-red)](https://www.uni-kiel.de/)

---

## What is CorpusDiff?

**CorpusDiff** is a computational linguistics pipeline that detects **value trends** in large text corpora over time. It was built for a research project analyzing **digital youth culture** using the [NottDeuYTSch](http://hdl.handle.net/11372/LRT-4806) corpus — 3 million YouTube comments from German-speaking adolescents (2008–2018).

CorpusDiff **correlates** linguistic patterns in a target corpus with **political value dimensions** extracted from a reference corpus (PoliTweets), revealing which values are rising or falling in youth discourse.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/CrazyDiamond-75/CorpusDiff.git
cd CorpusDiff/code

# Install dependencies (you'll need Python 3.9+)
# Hint: Install torch for CUDA or ROCm except you will suffer
pip install pandas numpy seaborn matplotlib scipy tqdm sentence_transformers torch torchvision

# Run CorpusDiff!
./corpusdiff.py main_corpus diff_corpus [huggingface_model]
```

> **Note:** You'll need the actual corpus data, e.g., `ndy_utf8.tsv` and `politweets_v02.csv`, to reproduce the paper results.
> These are not included in the repo!

---

## Main Results for CorpusDiff

| Dimension            | Correlation | Avg. Change per Quarter |
| -------------------- | ----------- | ----------------------- |
| Internationalism     | +0.166      | +0.044                  |
| Europe               | +0.158      | +0.040                  |
| Traditional Morality | -0.083      | -0.020                  |
| Constitution         | -0.083      | -0.023                  |

_Values are Pearson correlation coefficients with 95% confidence intervals_.

---

## CorpusDiff Pipeline

1. **Embedd** NottDeuYTSch and Politweets with RoBERTa
2. **Compare** NottDeuYTSch and PoliTweets value dimensions
3. **Calculate** _l_-metric for each positive and negative label in Politweets
4. **Generate** Pearson correlations and linear regression slopes
5. **Visualize** using boxenplots

The full methodology is documented in the accompanying thesis [`ha.tex`](ha.tex) _(PDF TO FOLLOW WHEN IT IS FINISHED)_

---

If you use CorpusDiff in your own research, please cite:

```bibtex
@unpublished{Hey26,
  author      = {Henri Heyden},
  title       = {Wertewandel in Digitaler Kinder- und Jugendkultur},
  institution = {Christian-Albrechts-Universität zu Kiel},
  year        = {2026},
  url         = {https://github.com/CrazyDiamond-75/CorpusDiff}
}
```
