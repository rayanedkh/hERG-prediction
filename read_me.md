Ce dossier contient de nombreux notebook contenant les démarches utilisés pour le Hackathon MARGO Telecom-AI Hackathon 2025.

Il contient d'une part les mmodèles brut optimisé grâce à watson.ai, la plateforme d'ibm.
- descriptors_xgb.ipynb : fait un XGBoost Classifier sur les 199 molecular features
- fingerprints_reg_log.ipynb, fingerprints_rf.ipynb, fingerprints_lgbm.ipynb : font respectivement une régression logistique, un random forest et un LGBM Classifier sur les vecteurs efcf et fcfc.
- ensemble.ipynb : un essai que j'avais fait pour rassembler les méthode mais que j'ai laissé tombé pour laisser place à un méta-modèle plus tard.

Le fichier task_1.ipynb m'as permi de comparer tous ces modèles ensembles, ainsi que d'autres que mes camarades m'ont fournis, notament un GCN et un transformer sur les smiles.
Dans le fichier meta_model.ipynb j'ai fait les ajustement finaux avant d'envoyer les predictions.

Le fichier task_2.ipynb contient toute ma démarche pour la task 2, les fonctions que j'utilise sont dans un fichier à part task_1_f.py. Globalement je calcul une matrice de similarité à partir de la distance de tanimoto, ensuite je regroupe en cluster qui ont une taille moyenne et minimum en tant qu'hyperparamètre. Je sépare les data en train et test comme d'habitude et j'utilise une cross validation qui respecte les cluster précedement calculés. Le tout dans un modèle Random Forest dont j'ai optimisé les paramètres à la main cette fois.

Les différents dossiers contiennent les fichiers intermédiaires enregistrés au cours des recherches, aisi que les data.