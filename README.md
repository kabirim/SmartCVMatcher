# SmartCVMatcher
SmartCVMatcher est un projet personnel basé sur le Deep Learning et le NLP (Natural Language Processing). Il permet d’automatiser l’évaluation de CVs par rapport à une offre d’emploi en utilisant un modèle Siamese Network entraîné sur des paires (CV, offre).
L’application fournit un score de compatibilité pour chaque CV, et peut classer plusieurs candidatures selon leur pertinence.
Elle est déployée via une interface simple en Streamlit, et utilise Sentence-BERT pour encoder les textes.

SmartCVMatcher est une application d’intelligence artificielle permettant de **comparer automatiquement des CVs à une offre d’emploi** et de **les classer par pertinence**. Ce projet combine le **Natural Language Processing (NLP)** avec des **Siamese Neural Networks** pour proposer un système intelligent d’aide au recrutement.

---

## 🎯 Objectifs

- Prédire si un CV correspond à une offre d'emploi
- Classer automatiquement plusieurs CVs en fonction de leur score de compatibilité
- Fournir une interface simple via **Streamlit**

---

## 🧠 Technologies utilisées

- **Python**
- **PyTorch** – pour le modèle Siamese
- **Sentence-BERT (all-MiniLM-L6-v2)** – pour l'encodage sémantique
- **scikit-learn** – pour la préparation des données
- **Streamlit** – pour l'interface utilisateur
- **Pandas, Torch, Joblib** – pour la gestion des données et la persistance

---

## Exemple d’utilisation
Offre d’emploi : Data Scientist avec expérience Python, NLP, Machine Learning

CVs : (extrait)
- CV 1 : Data Scientist 5 ans exp. Python, NLP
- CV 2 : Développeur mobile Flutter
- CV 3 : Data Analyst junior SQL/Excel

Résultat :
1. CV 1 (Score 0.92)
2. CV 3 (Score 0.61)
3. CV 2 (Score 0.21)

## Améliorations futures
- Support de fichiers PDF/Docx pour l’import des CVs
- Téléchargement des résultats en CSV
- Ajout d’un entraînement avec Triplet Loss pour un meilleur ranking
- Version API avec FastAPI

## 📄 Licence
Ce projet est libre d’utilisation à des fins personnelles ou éducatives. Pour un usage professionnel, merci de me contacter.
