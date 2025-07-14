# hERG-prediction

---

## About the project:

The following datasets are provided for the hackathon at Télécom Paris with the student association,  
taking place from 05/23/2025 to 05/25/2025.  
This event is organized by MARGO, Qubit-pharmaceuticals and IBM.

The aim of this event is to build a binary classifier capable of predicting whether a molecule is toxic or not.  
The toxicity studied here is that associated with hERG inhibition, cause of heart problems for certain drugs.

---

## Dataset Description

3 datasets are provided in the csv (comma-separated) format:

- **The training set**: `train.csv`  
    - 9415 rows, each corresponding to a molecule  
    - 1 `smiles` column, containing the chemical formula of the molecule represented by the row (in canonical SMILES format)  
    - 199 molecular features computed with the `rdkit` package  
    - 2048 columns (named `ecfc_XXXX`) containing the bit vector representation of Morgan fingerprints  
    - 2048 columns (named `fcfc_XXXX`) containing the bit vector representation of pharmacophore feature-based Morgan fingerprints  
    - 1 `class` column containing the label to predict (1 for hERG inhibitor, 0 otherwise)  

- **Test set 1**: `test_1.csv`  
    - 750 rows, each corresponding to a molecule  
    - 1 `smiles` column, containing the chemical formula of the molecule represented by the row (in canonical SMILES format)  
    - 199 molecular features computed with the `rdkit` package  
    - 2048 columns (named `ecfc_XXXX`) containing the bit vector representation of Morgan fingerprints  
    - 2048 columns (named `fcfc_XXXX`) containing the bit vector representation of pharmacophore feature-based Morgan fingerprints  

- **Test set 2**: `test_2.csv`  
    - 478 rows, each corresponding to a molecule  
    - 1 `smiles` column, containing the chemical formula of the molecule represented by the row (in canonical SMILES format)  
    - 1 `series` column, containing the identifier of the molecular series to which the molecule belongs  
    - 199 molecular features computed with the `rdkit` package  
    - 2048 columns (named `ecfc_XXXX`) containing the bit vector representation of Morgan fingerprints  
    - 2048 columns (named `fcfc_XXXX`) containing the bit vector representation of pharmacophore feature-based Morgan fingerprints  

All molecular features & fingerprints were generated using the [rdkit](https://www.rdkit.org/) python package version 2023.03.1.

---

## Copyright notice

Datasets used during the event were made available in the following article:  
Karim, A., Lee, M., Balle, T. et al. *CardioTox net: a robust predictor for hERG channel blockade based on  
deep learning meta-feature ensembles.* J Cheminform 13, 60 (2021). https://doi.org/10.1186/s13321-021-00541-z

Molecules in the datasets were sanitized using Qubit-pharmaceuticals preprocessing procedures.  
All molecular features were generated using the rdkit package (https://www.rdkit.org/).

---

## Présentation

Nous avons effectué ce projet en équipe de 3 (Merci à mes compagnons Noa Andre @noaac et Alexandre Mallez @AlexHibo) et sommes arrivés finaliste.

Ce hackathon était avant tout un moyen d'apprendre en s'amusant, en découvrant énormément de choses, de méthodes.  
De plus cette expérience nous a permis de développer notre coordination et la gestion d'un projet de ce type en équipe.

---

## Notre Travail

Notre travail se décompose en 2 parties.  
Ce dossier contient **5 sous-dossiers** principaux (utilisés pour les 2 parties) :

- `DNN_Descriptor`  
- `GNN`  
- `FP`  
- `Transformers`  
- `Final`  
- `Task 2`  
- `Annexe`  
- `Annexe 2`

---

### Partie 1

Dans le cadre de ce Hackathon, nous avons décidé pour les tâches 1 et 3 (détection sur un set de test uniforme) de combiner plusieurs modèles.

Ce premier travail s'effectuera sur les sous-dossiers :  
- `DNN_Descriptor`  
- `GNN`  
- `FP`  
- `Transformers`  
- `Final`  

Dans les datasets nous avions 3 types de features :  
- Descriptives  
- FingerPrints (FP)  
- SMILES  

En s'inspirant du papier de recherche *CardioTox net*, nous avons donc décidé de traiter ces 3 types de données séparément mais toujours avec un but final précis : combiner les forces de nos modèles sur chaque partie des données.

1) Pour les données **Descriptives**, nous avons entraîné un Multi Layer Perceptron (MLP) qui nous a permis d'obtenir des résultats cohérents (environ 80% accuracy comme les 3 modèles suivants).

2) Pour les données type **FingerPrints (FP)**, nous avons entraîné un XGBoost dont les hyperparamètres ont été optimisés avec Optuna.  
Nous avions un autre prétendant, le **Random Forest**, qui est assez intéressant notamment pour la réduction de la variance.  
Mais les résultats du XGBoost étaient légèrement meilleurs, ce qui nous a conduit vers ce choix.

3) Pour les données type **SMILES**, nous avons opté pour 2 stratégies assez différentes :  
   - Fine-tuner le modèle **seyonec/ChemBERTa-zinc-base-v1**, une adaptation de RoBERTa pour les données chimiques SMILES.  
     Nous avons utilisé `RobertaForSequenceClassification` de Hugging Face pour la classification.  
   - Construire un **GCNN (Graph Convolutional Neural Network)** pour traiter les molécules comme des graphes (atomes = nœuds, liaisons = arêtes).  
     Un GNN est donc une solution idéale pour ce type de données.

Pour la prédiction finale, on utilise les probabilités obtenues grâce aux 4 méthodes ci-dessus pour créer un méta-modèle :  
Nous avons essayé un **réseau de neurones dense**, mais les résultats n'étaient pas satisfaisants (probablement à cause d’un manque de données).  
Nous avons donc opté pour des méthodes plus traditionnelles : **moyenne**, **vote de majorité**, puis finalement une **régression logistique**.

---

### Partie 2

Ce second travail s'effectuera sur le sous-dossier :  
- `Task 2`

La tâche 2 nécessitait de faire de la prédiction sur des **séries (clusters de molécules)** isolées par rapport aux données d'entraînement.

Le premier réflexe a été de construire, à l'aide de **la mesure de Tanimoto**, un jeu d'entraînement ressemblant à celui de test.  
Beaucoup d'ajustements ont dû être faits : comment perdre le moins de data possible, combien de cellules par cluster dans le train...  
C'est à travers de nombreux tests que cela a pu se résoudre (voir `Annexe 2` pour certains détails).

Une fois ce nouveau jeu d'entraînement construit, il nous a suffi d'essayer et de comparer des modèles simples de la bonne manière.  
Finalement, c'est un **Random Forest** bien paramétré qui nous a permis d'avoir les meilleurs résultats sur cette tâche.

---

## Bonus

Dans les dossiers `Annexe 1` et `Annexe 2` figurent un lot d'essais qui n'ont pas été conservés pour le résultat final,  
mais qui nous ont permis de mieux comprendre les données et d'agir en conséquence.  
Pour la plupart, cela consiste en réseaux de neurones non aboutis (sur GCNN ou simple DNN), ou en méthodes simples comme XGBoost, RandomForest ou encore LightGBM.  
Il peut être intéressant de les regarder pour mieux comprendre comment notre projet a évolué, et pourquoi nous avons fait certains choix au lieu d'autres.
