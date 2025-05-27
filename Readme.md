Ce dossier contient **5 sous-dossiers**:

- DNN_Descriptor
- GNN
- FP
- Transformers
- Final

Dans le cadre de ce Hackathon, nous avons décidé pour les tâches 1 et 3 de combiner plusieurs modèles.

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

Ce hackathon était avant tout un moyen d'apprendre en s'ammusant, nous avons donc décidé de faire les 2 approches et d'aller jusqu'au bout dans l'objectif que les 2 soient complémentaire sur les data.

A la fin, on utilise les probabilités obtenues grace au 4 méthodes ci-dessus pour créer un méta-modèle et obtenir une nouvelle probabilité qui nous permmetra déjà de classer pour la task3, puis ensuite pour la task 1.


