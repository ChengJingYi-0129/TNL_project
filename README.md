---
title: Cinema Sentiment Analysis
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---

# Cinema Sentiment Analysis

A Malaysia-focused NLP application that classifies Google Maps cinema reviews as **positive**, **neutral**, or **negative**, then presents results through an interactive Gradio interface.

**Live demo:** https://huggingface.co/spaces/CJY0129/cinema-sentiment-analysis

## What It Does

- Analyses one review at a time for quick feedback
- Imports CSV files for batch analysis
- Visualises sentiment distributions and summaries
- Supports review-result and flagged-data management
- Uses a trained BiGRU sentiment model for predictions

## Tech Stack

- Python
- Gradio
- TensorFlow / Keras
- Pandas and scikit-learn
- Hugging Face Spaces

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Dataset

The project uses selected Google Maps reviews of GSC cinema branches in Kuala Lumpur and Melaka for academic analysis. Handle source data responsibly and do not add personally identifiable information to the repository.

## Repository Layout

- `app.py` — Gradio application
- `best_sentiment_bigru/` — trained model artefact
- `data/` — local project data
- `notebook/` — experimentation and analysis notebooks

## Project Context

Built as an academic NLP project to turn unstructured customer feedback into an accessible sentiment-analysis workflow.