# hERG-prediction

---

About the project:

The following datasets are provided for the hackathon at Ecole Telecom Paris with the student association,
taking place from 05/23/2025 to 05/25/2025.
This event is organized by MARGO, Qubit-pharmaceuticals and IBM.

The aim of this event is to build a binary classifier capable of predicting whether a molecule is toxic or not.
The toxicity studied here is that associated with hERG inhibition, cause of heart problems for certain drugs.

====================
DATASETS DESCRIPTION
====================

3 datasets are provided in the csv (comma-separated) format:

- The training set: 'train.csv'
    - 9415 rows, each corresponding to a molecule
    - 1 'smiles' column, containing the chemical formula of the molecule represented by the row (in canonical SMILES format)
    - 199 molecular features computed with the rdkit package
    - 2048 columns (named 'ecfc_XXXX') containing the bit vector representation of Morgan fingerprints
    - 2048 columns (named 'fcfc_XXXX') containing the bit vector representation of pharmacophore feature-based Morgan fingerprints
    - 1 'class' column containing the label to predict (1 for hERG inhibitor, 0 otherwise)

- Test set 1: 'test_1.csv'
    - 750 rows, each corresponding to a molecule
    - 1 'smiles' column, containing the chemical formula of the molecule represented by the row (in canonical SMILES format)
    - 199 molecular features computed with the rdkit package
    - 2048 columns (named 'ecfc_XXXX') containing the bit vector representation of Morgan fingerprints
    - 2048 columns (named 'fcfc_XXXX') containing the bit vector representation of pharmacophore feature-based Morgan fingerprints

- Test set 2: 'test_2.csv'
    - 478 rows, each corresponding to a molecule
    - 1 'smiles' column, containing the chemical formula of the molecule represented by the row (in canonical SMILES format)
    - 1 'series' columns, containing the identifier of the molecular series to which the molecule belongs
    - 199 molecular features computed with the rdkit package
    - 2048 columns (named 'ecfc_XXXX') containing the bit vector representation of Morgan fingerprints
    - 2048 columns (named 'fcfc_XXXX') containing the bit vector representation of pharmacophore feature-based Morgan fingerprints


All molecular features & fingerprints were generated using the rdkt (https://www.rdkit.org/) python package version 2023.03.1.

================
COPYRIGHT NOTICE
================

Datasets used during the event were made available in the following Article.
Karim, A., Lee, M., Balle, T. et al. CardioTox net: a robust predictor for hERG channel blockade based on
deep learning meta-feature ensembles. J Cheminform 13, 60 (2021). https://doi.org/10.1186/s13321-021-00541-z

Molecules in the datasets were sanitized using Qubit-pharmaceuticals preprocessing procedures.
All molecular features were generated using the rdkit package (https://www.rdkit.org/).

---
---

# Présentation

Nous avons effectué ce projet en équipe de 3 (Merci à mes compagnons Noa Andre et Rayane Dakhlaoui) et sommes arrivés finaliste.

Ce hackathon était avant tout un moyen d'apprendre en s'ammusant, nous avons donc décidé de faire les 2 approches et d'aller jusqu'au bout dans l'objectif que les 2 soient complémentaire sur les data.


## Notre Travail

Ce dossier contient **5 sous-dossiers**:

- DNN_Descriptor
- GNN
- FP
- Transformers
- Final

Dans le cadre de ce Hackathon, nous avons décidé pour les tâches 1 et 3 (détéction sur un set de test uniforme) de combiner plusieurs modèles.

Dans les dataset nous avions 3 types de features
- Descriptives
- FingerPrints (FP)
- SMILES

Nous avons donc décidé de traiter ces 3 types de données séparemment mais toujours avec un but final précis: Combiner les forces de nos modèles.


Pour les données **Descriptives**, nous avons entraîné un Multi Layer Perceptron (MLP) qui nous a permis d'obtenir les résultats suivant:
**Cross-validation results:**
- Mean Accuracy: **0.8050 (±0.0050)**
- Mean Kappa: **0.6098 (±0.0102)**

Pour les données type **FingerPrints (FP)**, nous avons entrainé un XGBoost dont les hyperparamètres ont été optimisé avec Optuna. Nous avions un autre prétedant, le **Random Forrest** qui est assez intéressant notamment pour la réduction de la variance. Mais les résultats du XGBoost étaient légèrement meilleurs, ce qui nous a conduit vers ce choix.
Le XGBoost avec Optuna modèle nous a permis d'obtenir les résultats suivant:

- Cohen's Kappa: **0.6387962055039826**
- Accuracy: **0.8194370685077005**

Pour les données type **SMILES**, nous avons opté pour 2 startégies asez différentes. La première était de fine-tuner le modèle **seyonec/ChemBERTa-zinc-base-v1**, qui est une adaptation de RoBERTa (qui est lui même un modèle de type **Transformer** initialement conçu pour le langage naturel) qui a été entrainé non pas sur du texte classique, mais sur des représentations chimiques de molécules au format SMILES. L'objectif était donc claire finetuner un modèle capable de comprendre les données au format SMILES dans l'objectif d'une classification binaire. Pour la tokenization, nous nous sommes basés sur la même source. EN effet, nous avons utilisé **RobertaForSequenceClassification** qui est une classe de Hugging Face Transformers qui ajoute une tête de classification (typiquement une couche linéaire) au-dessus du modèle RoBERTa.

La deuxième approche était de faire un **GCNN (Graph Convultional NN)** qui permet donc de traiter d'une assez belle manière la chimie moléculaire car dans ce cadre les atomes sont des noeuds et leurs laisons sont les arêtes du graph. Un GNN était donc une solution idéale pour ce type de data et ce type de prblématique de classification binaire.

A la fin, on utilise les probabilités obtenues grace au 4 méthodes ci-dessus pour créer un méta-modèle et obtenir une nouvelle probabilité qui nous permmetra déjà de classer pour la task3, puis ensuite pour la task 1. Nous avons essayé d'abord avec un réseau de neurone dense comme nous suggérait certains papier de recherche, cependant les résultats n'étant pas au rendez-vous sur notre set de validation (surement due à un manque de data). Ainsi nous avons plutôt obpté pour des méthodes plus traditionnels comme en prenant simplement la moyenne, un vote de majorité, puis finallement avec une régression logistique.







