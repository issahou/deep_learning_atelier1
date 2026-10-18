# Atelier 1 – Deep Learning avec PyTorch (DNN / MLP)

**Université Abdelmalek Essaadi – Faculté des Sciences et Techniques de Tanger – Département Génie Informatique**
Master MBD · Module Deep Learning · Pr. ELAACHAK Lotfi

**Objectif :** se familiariser avec PyTorch pour réaliser une tâche de **régression** (Partie 1) et une tâche de **classification multi-classes** (Partie 2) avec des architectures DNN/MLP.

---

## 1. Contenu du dépôt

| Fichier | Description |
|---|---|
| `Atelier1_Partie1_Regression_NYSE.ipynb` | Partie 1 : régression sur le dataset NYSE |
| `Atelier1_Partie2_Classification_Maintenance.ipynb` | Partie 2 : classification multi-classes (maintenance prédictive) |
| `results/part1_regression_results.csv` | Tableau comparatif exporté par le notebook 1 |
| `results/part2_classification_results.csv` | Tableau comparatif exporté par le notebook 2 |
| `requirements.txt` | Dépendances Python |

## 2. Exécution

1. **Kaggle :** importer le notebook, ajouter le dataset via *Add Data*, puis *Run All*.
2. **Google Colab :** importer le notebook et déposer le CSV dans `/content/` (ou laisser le notebook le télécharger avec `kagglehub`), puis *Tout exécuter*.
3. **En local :** placer le CSV à côté du notebook (ou dans `data/`) et installer `requirements.txt`.

Les cellules doivent être exécutées **dans l'ordre**, en commençant par celle des imports. Graine aléatoire fixée (`42`) ; de légères variations restent possibles selon le matériel (CPU/GPU).

---

## 3. Partie 1 – Régression (NYSE)

### 3.1 Données et préparation

- **Dataset :** `prices-split-adjusted.csv` (Kaggle `dgawlik/nyse`) : 851 264 lignes, 501 actions, du 2010-01-04 au 2016-12-30. Aucune valeur manquante, aucun doublon.
- **Tâche :** prédire le cours de clôture (`close`) **du jour suivant** de chaque action.
- **Variables (9)** : `log_open`, `log_high`, `log_low`, `log_close`, `log_ma5`, `log_ma10`, `ret_1`, `vol5`, `log_volume`, toutes calculées par action à partir du passé et du jour courant (pas de fuite d'information). Cible : `log(close du lendemain)`.
- **Pourquoi le logarithme :** les cours vont de 1.59 $ à 1 578 $ ; le volume est très asymétrique (médiane 2.5 M, maximum 860 M).
- **Découpage chronologique** (par dates) : 70 % train / 10 % validation / 20 % test. Un découpage aléatoire serait trop optimiste pour une série temporelle. Standardisation ajustée sur le train uniquement.
- **« Précision » en régression :** part des prédictions dont l'erreur relative est ≤ 2 % (`Acc±2%`), en plus de RMSE, MAE (en $) et R².
- **Référence naïve :** « le cours de demain = le cours d'aujourd'hui ». C'est le point de comparaison indispensable.

### 3.2 Recherche d'hyperparamètres (`GridSearchCV`)

Grille de 24 combinaisons × 3 plis (`TimeSeriesSplit`) = 72 entraînements, sur un sous-échantillon de 20 000 lignes du train (ordre chronologique conservé). Score : R².

| Hyperparamètre | Valeurs testées |
|---|---|
| Architecture | `(32,)`, `(64, 32)`, `(128, 64, 32)` |
| Learning rate | 0.001, 0.01 |
| Optimiseur | Adam, SGD (momentum 0.9) |
| Époques | 20, 40 |

**Meilleure configuration :** `(128, 64, 32)`, lr = 0.001, Adam, 40 époques (R² de validation = 0.99926).

**À retenir :** les 10 meilleures configurations ont un R² compris entre 0.99916 et 0.99926, soit moins d'écart que l'écart-type entre plis (≈ 0.00027). Le choix de la « meilleure » configuration n'est donc **pas statistiquement significatif** : l'architecture, le learning rate et le nombre d'époques ont très peu d'influence ici. Seul constat net : Adam occupe les 7 premières places, SGD apparaît au 8ᵉ rang.

### 3.3 Modèle de base : courbes Perte/Époques et Précision/Époques

| | Train | Test |
|---|---|---|
| Perte finale (MSE, cible standardisée) | 0.0007 | 0.0008 |
| Précision ±2 % | 81.0 % | 78.8 % |

Interprétation : les courbes de perte train et test restent proches et se stabilisent (la perte test atteint son minimum vers l'époque 26 sur 40, puis plafonne). **Il n'y a pas de sur-apprentissage** et la perte est faible : le modèle apprend correctement. Les 40 époques sont un peu plus que nécessaire.

### 3.4 Régularisation et comparaison

Cinq variantes, avec les hyperparamètres du modèle de base : L2 (weight decay 1e-3), Dropout (0.2), Batch Normalization, early stopping (patience 6, sur la validation), et une combinaison (Dropout 0.2 + L2 1e-4 + BatchNorm + early stopping).

| Modèle | RMSE train ($) | RMSE test ($) | MAE test ($) | R² test | Acc±2 % train | Acc±2 % test |
|---|---|---|---|---|---|---|
| **Référence naïve** | 1.384 | **2.212** | 0.993 | 0.9995 | 82.3 % | 80.4 % |
| Modèle de base | 1.561 | 2.689 | 1.120 | 0.9993 | 81.0 % | 78.8 % |
| L2 (1e-3) | 1.689 | 3.153 | 1.173 | 0.9991 | 78.0 % | 77.0 % |
| Dropout (0.2) | 13.652 | 27.900 | 7.276 | 0.9258 | 47.8 % | 38.4 % |
| BatchNorm | 3.256 | 6.622 | 1.943 | 0.9958 | 77.0 % | 71.0 % |
| Early stopping | 1.417 | 2.270 | 1.020 | 0.9995 | 81.6 % | 79.7 % |
| Combiné | 2.840 | 5.498 | 1.937 | 0.9971 | 52.5 % | 57.6 % |

(L'early stopping s'arrête à l'époque 17, avec les meilleurs poids de l'époque 11 ; le modèle combiné s'arrête à l'époque 12, meilleure époque 6.)

### 3.5 Interprétation des résultats

1. **Le R² n'est pas informatif ici.** Il vaut 0.9993 pour le modèle et 0.9995 pour la référence naïve : le cours de demain est presque identique à celui d'aujourd'hui, donc n'importe quel modèle raisonnable obtient un R² proche de 1.
2. **Aucun modèle ne bat la référence naïve** sur la RMSE, la MAE et la précision ±2 %. Le plus proche est l'early stopping (RMSE test 2.270 $ contre 2.212 $, soit +2.6 %) ; le modèle de base est 21.6 % moins bon. Le réseau n'apporte donc pas d'information supplémentaire par rapport à la simple persistance du cours.
3. **Pas de sur-apprentissage, donc peu d'intérêt à régulariser.** Le `Δ R²` (train − test) est quasi nul (≤ 0.0013) pour tous les modèles sauf Dropout. La régularisation réduit la variance, mais ici la difficulté est la précision : prédire le cours de demain revient presque à recopier `close`, et le bruit ajouté par Dropout ou BatchNorm dégrade probablement cette précision.
4. **Dropout (0.2), BatchNorm et le modèle combiné dégradent fortement les résultats.** Dropout a une erreur élevée *aussi sur le train* (RMSE 13.65 $, précision 47.8 %) : c'est du **sous-apprentissage**, non du sur-apprentissage. *(Le message de diagnostic automatique du notebook indique « sur-apprentissage » pour ces cas, car il compare seulement train et test ; l'analyse des erreurs train montre l'inverse.)*
5. **L'early stopping est la seule technique qui n'abîme pas le modèle** : il améliore même le modèle de base (RMSE test 2.270 $ contre 2.689 $) en s'arrêtant à l'époque 11. Une seule exécution par configuration est disponible : une part de cet écart peut venir de l'aléa de l'entraînement.
6. **La RMSE en dollars augmente du train au test pour tous les modèles, y compris la référence naïve** (1.38 → 2.21). Ce n'est pas du sur-apprentissage : il s'agit vraisemblablement d'un effet d'échelle : la RMSE en dollars est dominée par les actions chères (jusqu'à 1 578 $) et varie avec le niveau des prix de la période considérée.

### 3.6 Limites et pistes

- Une seule partition chronologique et une seule graine ; pas d'intervalle de confiance.
- Prédire le **rendement** (`close_{t+1}/close_t − 1`) ou le **résidu** par rapport au cours d'aujourd'hui, plutôt que le niveau de prix, forcerait le réseau à partir de la référence naïve et rendrait la comparaison plus significative.
- Pistes : fenêtres temporelles plus longues, LSTM/GRU, intégration de `fundamentals.csv`.

---

## 4. Partie 2 – Classification multi-classes (maintenance prédictive)

### 4.1 Prétraitement et EDA

- **Dataset :** `predictive_maintenance.csv` (Kaggle `shivamb/machine-predictive-maintenance-classification`) : 10 000 lignes, aucune valeur manquante, aucun doublon.
- **Nettoyage :** suppression des identifiants (`UDI`, `Product_ID`) et de **27 lignes aux étiquettes contradictoires** : 9 lignes avec `Target = 1` mais « No Failure », et 18 lignes avec `Target = 0` mais une panne déclarée (ce sont toutes les lignes « Random Failures »). Il reste **9 973 lignes et 5 classes**.
- **Encodage :** `Type` (L < M < H) en ordinal ; variables dérivées `temp_diff` (écart de température), `power_W` (puissance mécanique) et `wear_x_torque` (usure × couple).
- **Valeurs aberrantes (IQR) :** 418 pour la vitesse de rotation, 70 pour le couple, 0 pour les autres. Elles sont **conservées** car elles correspondent souvent aux pannes.
- **Déséquilibre :** après nettoyage, « No Failure » = 96.69 %, Heat Dissipation 1.12 %, Power 0.95 %, Overstrain 0.78 %, Tool Wear 0.45 %. L'accuracy seule est donc trompeuse (prédire toujours « No Failure » donne déjà ≈ 96.7 %). On juge le modèle sur le **F1 macro** et la **sensibilité (recall) par classe**.

### 4.2 Découpage, standardisation, augmentation

- Découpage stratifié : **72 % train / 8 % validation / 20 % test** (test : 1 995 lignes, dont seulement 9 à 22 par classe minoritaire).
- Standardisation ajustée sur le train uniquement.
- **SMOTE** appliqué **au train seulement** pour équilibrer les 5 classes ; validation et test gardent la distribution réelle.
- Dans `GridSearchCV`, SMOTE est placé dans un `imblearn.Pipeline` : il n'est appliqué qu'aux plis d'entraînement (aucune fuite de points synthétiques vers la validation).

### 4.3 Recherche d'hyperparamètres

Grille de 24 combinaisons × 3 plis stratifiés = 72 entraînements ; score : **F1 macro**.

| Hyperparamètre | Valeurs testées |
|---|---|
| Architecture | `(32,)`, `(64, 32)`, `(128, 64)` |
| Learning rate | 0.001, 0.01 |
| Optimiseur | Adam, SGD (momentum 0.9) |
| Époques | 20, 40 |

**Meilleure configuration :** `(128, 64)`, lr = 0.01, Adam, 20 époques (F1 macro de validation croisée = 0.686). Adam domine nettement le classement (SGD apparaît au 10ᵉ rang avec 0.620).

### 4.4 Modèle de base : courbes et métriques

Courbes (train mesuré sur le train réel, non augmenté) : perte finale train 0.104, test 0.250 ; accuracy train 97.1 %, test 95.1 %. La perte test atteint son minimum à l'époque 13 sur 20 puis remonte : **sur-apprentissage modéré**.

| Modèle | Acc train | Acc test | Recall macro train | Recall macro test | F1 macro train | F1 macro test | Δ F1 (train−test) | AUC test |
|---|---|---|---|---|---|---|---|---|
| Base | 0.9712 | 0.9514 | 0.9905 | 0.7446 | 0.7511 | 0.6043 | 0.1468 | 0.9738 |
| L2 (1e-3) | 0.9370 | 0.9258 | 0.9810 | 0.8753 | 0.6242 | 0.5920 | 0.0323 | 0.9847 |
| Dropout (0.3) | 0.9409 | 0.9308 | 0.9818 | 0.8399 | 0.6377 | 0.5941 | 0.0436 | 0.9770 |
| BatchNorm | 0.9777 | 0.9574 | 0.9954 | 0.7623 | 0.8118 | **0.6583** | 0.1535 | 0.9710 |
| Label smoothing (0.1) | 0.9830 | 0.9644 | 0.9965 | 0.7354 | 0.8264 | 0.6382 | 0.1882 | 0.9692 |
| Early stopping | 0.9837 | 0.9684 | 0.9904 | 0.7585 | 0.8236 | 0.6578 | 0.1658 | 0.9767 |
| Combiné | 0.9235 | 0.9113 | 0.9813 | 0.8944 | 0.6240 | 0.5867 | 0.0373 | 0.9855 |

(Early stopping : arrêt à l'époque 21, meilleure époque 13 ; combiné : arrêt à l'époque 25, meilleure époque 17.)

### 4.5 Meilleur modèle selon le F1 macro test : BatchNorm

| Métrique | Train | Test |
|---|---|---|
| Accuracy | 0.9777 | 0.9574 |
| Balanced accuracy | 0.9954 | 0.7623 |
| Précision (macro) | 0.7414 | 0.5993 |
| Sensibilité / Recall (macro) | 0.9954 | 0.7623 |
| Spécificité (macro) | 0.9955 | 0.9563 |
| F1 (macro) | 0.8118 | 0.6583 |
| F1 (pondéré) | 0.9832 | 0.9675 |
| AUC ROC (OvR, macro) | 0.9997 | 0.9710 |

Rapport par classe sur le test :

| Classe | Précision | Recall | F1 | Effectif |
|---|---|---|---|---|
| Heat Dissipation Failure | 0.690 | 0.909 | 0.784 | 22 |
| No Failure | 0.994 | 0.963 | 0.978 | 1 929 |
| Overstrain Failure | 0.700 | 0.875 | 0.778 | 16 |
| Power Failure | 0.571 | 0.842 | 0.681 | 19 |
| Tool Wear Failure | 0.042 | 0.222 | 0.070 | 9 |

### 4.6 Interprétation des résultats

1. **L'accuracy masque l'essentiel.** Elle dépasse 91 % pour tous les modèles alors que le F1 macro n'atteint que 0.59 à 0.66 : la performance sur les pannes rares est bien plus faible que celle sur « No Failure ».
2. **Sur-apprentissage marqué du modèle de base** : recall macro 0.99 en train contre 0.74 en test, F1 macro 0.75 contre 0.60. Le réseau mémorise les rares exemples de pannes (et leurs voisins synthétiques SMOTE).
3. **Effet du rééquilibrage :** même sur le train réel, la précision macro (0.74) est bien inférieure au recall (0.995). Entraîné sur des classes équilibrées, le réseau déclare trop souvent une panne sur des machines saines : le recall monte, la précision baisse (précision test 0.60).
4. **Compromis précision/recall selon la régularisation.** L2, Dropout et le modèle combiné réduisent fortement l'écart train/test (Δ F1 de 0.03 à 0.04 contre 0.15 pour le modèle de base) et augmentent le recall macro test (0.84 à 0.89) et l'AUC (jusqu'à 0.986), mais au prix d'une précision plus faible (accuracy de 91 % à 93 %, F1 macro ≈ 0.59). BatchNorm et early stopping obtiennent le meilleur F1 macro test (0.658 chacun) avec un recall plus faible (0.76).
5. **Quel modèle choisir ?** En maintenance, manquer une panne coûte généralement plus cher qu'une fausse alerte : L2 ou le modèle combiné (recall 0.88 à 0.89) sont préférables. Si l'on veut un équilibre précision/recall, BatchNorm ou early stopping.
6. **La classe « Tool Wear Failure » n'est pas apprise** (F1 = 0.07, précision 0.04, recall 0.22 sur 9 exemples). Elle n'a que 45 exemples réels (environ 32 dans le train) : SMOTE interpole entre très peu de points et ne crée pas d'information nouvelle. Il est probable que ces pannes soient peu séparables des machines saines avec les variables disponibles.
7. **Les écarts entre variantes sont à interpréter avec prudence.** Le test contient 9 à 22 exemples par classe minoritaire : une seule erreur de classification change le recall d'une classe de 5 à 11 points. Les différences de F1 macro test (0.587 à 0.658) ne permettent pas de désigner un vainqueur certain.

### 4.7 Limites et pistes

- Une seule partition train/test et une seule graine : une validation croisée répétée donnerait des résultats plus fiables.
- Aucune expérience **sans SMOTE** : sa contribution propre n'est pas mesurée.
- La classe « Random Failures » a été retirée (étiquettes incohérentes).
- Pistes : pondération des classes ou *focal loss*, autres méthodes d'augmentation (ADASYN, SMOTE-ENN), réglage du seuil de décision, comparaison avec des arbres de gradient (XGBoost).

---

## 5. Conclusion générale

- Les deux notebooks suivent une démarche complète (EDA, DNN PyTorch, `GridSearchCV`, courbes, régularisation, comparaison) et évitent les fuites de données : découpage chronologique, standardisation ajustée sur le train, SMOTE limité au train et intégré à la validation croisée.
- **Partie 1 :** le MLP converge sans sur-apprentissage mais **ne bat pas la référence naïve** ; seule la technique d'early stopping s'en rapproche. La régularisation n'apporte rien quand le problème n'est pas un problème de variance.
- **Partie 2 :** le MLP détecte correctement les pannes de type Heat Dissipation, Overstrain et Power (recall 0.84 à 0.91) mais pas Tool Wear. La régularisation réduit nettement le sur-apprentissage, au prix d'une baisse de précision.

---

## 6. Synthèse personnelle : ce que j'ai appris

- **PyTorch :** construire un MLP (couches, activations, Dropout, BatchNorm), écrire la boucle d'entraînement (perte, rétropropagation, optimiseur) et choisir la perte adaptée à la tâche (MSE pour la régression, entropie croisée avec Softmax pour la classification).
- **Intégration avec scikit-learn :** encapsuler un modèle PyTorch dans un estimateur (`fit`, `predict`, `score`) pour utiliser `GridSearchCV`, et constater que les hyperparamètres n'ont pas toujours un effet significatif (partie 1 : écarts inférieurs à l'écart-type).
- **Éviter les fuites de données :** découpage chronologique pour une série temporelle ; standardisation ajustée sur le train ; SMOTE appliqué uniquement au train et placé dans un pipeline pour la validation croisée.
- **Choisir de bonnes métriques et de bonnes références :** un R² de 0.999 n'a aucune valeur sans comparaison avec une référence naïve (ici, elle fait mieux que mon réseau) ; sur données déséquilibrées, l'accuracy est trompeuse et il faut regarder le F1 macro, le recall par classe et la matrice de confusion.
- **Lire les courbes d'apprentissage :** distinguer sur-apprentissage (écart train/test) et sous-apprentissage (erreur élevée aussi sur le train). La régularisation n'est utile que s'il y a un problème de variance : Dropout et BatchNorm ont dégradé la régression, alors qu'en classification ils ont réduit le sur-apprentissage.
- **Les limites de l'augmentation de données :** SMOTE aide à détecter les pannes rares, mais augmente les fausses alertes et ne peut pas compenser une classe de 45 exemples (Tool Wear).
- **Esprit critique sur les résultats :** avec 9 à 22 exemples par classe minoritaire en test et une seule graine, il faut éviter de conclure trop vite sur de petits écarts.
