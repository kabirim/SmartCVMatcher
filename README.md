# SmartCVMatcher

An AI-powered recruitment assistant that automatically evaluates and ranks resumes against job descriptions using **Natural Language Processing (NLP)** and **Machine Learning**.

---

# Overview

SmartCVMatcher is a personal project designed to automate the resume screening process by measuring the semantic similarity between a candidate's resume and a job description.

The application combines **Sentence-BERT embeddings** with an **XGBoost classifier** to predict whether a candidate is a good match for a position and generates a compatibility score that can be used to rank multiple applicants.

A lightweight **Streamlit** interface allows users to easily upload resumes and compare them against a job offer.

---

# Features

* Resume and job description semantic comparison
* Automatic candidate matching
* Compatibility score prediction
* Resume ranking based on relevance
* Interactive Streamlit interface
* Deep Learning and NLP-based semantic understanding

---

# Project Architecture

```text
Job Description + Resume(s)
            │
            ▼
Sentence-BERT Embedding
            │
            ▼
Feature Vector Generation
            │
            ▼
XGBoost Classifier
            │
            ▼
Compatibility Score
            │
            ▼
Candidate Ranking
```

---

# Technology Stack

## Backend

* Python

## Machine Learning

* XGBoost (XGBClassifier)
* scikit-learn

## Natural Language Processing

* Sentence-BERT (`all-mpnet-base-v2`)

## Frontend

* Streamlit

## Data Processing

* Pandas
* Joblib

---

# Model Pipeline

## 1. Text Encoding

Both the job description and resumes are converted into dense semantic vectors using the pretrained Sentence-BERT model:

```text
all-mpnet-base-v2
```

This approach captures contextual meaning instead of relying solely on keyword matching.

---

## 2. Candidate Classification

The generated embeddings are processed by an **XGBoost classifier**, trained to determine whether a resume is relevant to a given job description.

The model outputs a compatibility probability between **0 and 1**.

---

## 3. Candidate Ranking

After computing compatibility scores, resumes are automatically sorted from the most relevant candidate to the least relevant one.

This ranking system enables recruiters to quickly identify the best profiles.

---

# Example

## Job Description

```text
Data Scientist with experience in Python,
Natural Language Processing and Machine Learning.
```

## Candidate Resumes

```text
Resume 1
Data Scientist with 5 years of experience
in Python and NLP.
```

```text
Resume 2
Mobile Developer specialized in Flutter.
```

```text
Resume 3
Junior Data Analyst with SQL and Excel skills.
```

## Ranking Result

| Rank | Candidate | Compatibility Score |
| ---- | --------- | ------------------- |
| 1    | Resume 1  | **0.92**            |
| 2    | Resume 3  | **0.61**            |
| 3    | Resume 2  | **0.21**            |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/kabirim/SmartCVMatcher.git

cd SmartCVMatcher
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# Project Objectives

* Automate resume screening
* Reduce manual candidate evaluation
* Improve recruitment efficiency using AI
* Demonstrate the integration of NLP and Machine Learning in a real-world application
* Provide an intuitive interface for recruiters and HR professionals

---

# Future Improvements

Planned features include:

* PDF and DOCX resume parsing
* CSV export for ranking results
* Triplet Loss training for improved semantic ranking
* FastAPI deployment for production environments
* Batch processing of large resume collections
* Advanced analytics and candidate insights

---

# License

This project is available for personal and educational use.

For commercial or professional use, please contact the author.
