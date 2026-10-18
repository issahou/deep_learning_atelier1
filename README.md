
**Objectif :** utiliser PyTorch pour une tâche de régression (Partie 1) et une tâche de classification multi-classes (Partie 2).

## Contenu

| Fichier | Description |
|---|---|
| `Atelier1_Partie1_Regression_NYSE.ipynb` | Partie 1 : régression (NYSE) |
| `Atelier1_Partie2_Classification_Maintenance.ipynb` | Partie 2 : classification (maintenance prédictive) |


**Exécution :** Kaggle ou Colab, avec le CSV du dataset, puis *Run All* (cellules dans l'ordre, en commençant par les imports). Graine fixée à 42.

---

## Partie 1 – Régression (NYSE)

**Données :** `prices-split-adjusted.csv` (851 264 lignes, 501 actions, 2010–2016).

**Méthode :**
- Variables calculées par action à partir du passé (prix en log, moyennes mobiles, rendement, volatilité, volume).
- Découpage **chronologique** 70/10/20 %, standardisation ajustée sur le train.
- `GridSearchCV` (24 combinaisons : architecture, learning rate, optimiseur, époques). Meilleur choix : `(128, 64, 32)`, lr 0.001, Adam, 40 époques. Les 10 meilleures configurations sont à moins de 0.0001 de R² : le choix est peu significatif.
- « Précision » = part de prédictions à ±2 % du vrai cours. Référence naïve : cours de demain = cours d'aujourd'hui.

**Résultats  :**

| Modèle | RMSE ($) | R² | Précision ±2 % |
|---|---|---|---|
| Référence naïve | **2.212** | 0.9995 | 80.4 % |
| Modèle de base | 2.689 | 0.9993 | 78.8 % |
| L2 | 3.153 | 0.9991 | 77.0 % |
| Dropout (0.2) | 27.900 | 0.9258 | 38.4 % |
| BatchNorm | 6.622 | 0.9958 | 71.0 % |
| Early stopping | 2.270 | 0.9995 | 79.7 % |
| Combiné | 5.498 | 0.9971 | 57.6 % |

**Interprétation :**
- Les courbes de perte train et test sont proches et se stabilisent : pas de sur-apprentissage.
- Le R² est peu informatif (0.9995 pour la référence naïve). **Aucun modèle ne bat la référence naïve** ; l'early stopping s'en approche le plus.
- La régularisation n'aide pas ici, faute de sur-apprentissage. Dropout et le modèle combiné **sous-apprennent** (erreur élevée aussi sur le train).

---

## Partie 2 – Classification (maintenance prédictive)

**Données :** `predictive_maintenance.csv` (10 000 lignes). **Cible :** `Failure Type`.

**Méthode :**
- Nettoyage : suppression des identifiants et de 27 lignes aux étiquettes contradictoires (dont toute la classe « Random Failures ») : **9 973 lignes, 5 classes**.
- Classes très déséquilibrées : « No Failure » = 96.7 %. On juge donc le F1 macro et le recall, pas l'accuracy seule.
- Découpage stratifié 72/8/20 %, standardisation sur le train, **SMOTE sur le train uniquement** (et dans le pipeline du `GridSearchCV` pour éviter les fuites).
- `GridSearchCV` (24 combinaisons, score F1 macro). Meilleur choix : `(128, 64)`, lr 0.01, Adam, 20 époques.

**Résultats (test) :**

| Modèle | Accuracy | Recall macro | F1 macro | Δ F1 (train−test) |
|---|---|---|---|---|
| Base | 0.9514 | 0.7446 | 0.6043 | 0.147 |
| L2 | 0.9258 | 0.8753 | 0.5920 | 0.032 |
| Dropout (0.3) | 0.9308 | 0.8399 | 0.5941 | 0.044 |
| BatchNorm | 0.9574 | 0.7623 | **0.6583** | 0.154 |
| Label smoothing | 0.9644 | 0.7354 | 0.6382 | 0.188 |
| Early stopping | 0.9684 | 0.7585 | 0.6578 | 0.166 |
| Combiné | 0.9113 | 0.8944 | 0.5867 | 0.037 |

**Meilleur modèle (BatchNorm), par classe sur le test :**

| Classe | Précision | Recall | F1 | Effectif |
|---|---|---|---|---|
| Heat Dissipation | 0.690 | 0.909 | 0.784 | 22 |
| No Failure | 0.994 | 0.963 | 0.978 | 1 929 |
| Overstrain | 0.700 | 0.875 | 0.778 | 16 |
| Power | 0.571 | 0.842 | 0.681 | 19 |
| Tool Wear | 0.042 | 0.222 | 0.070 | 9 |

**Interprétation :**
- Le modèle de base sur-apprend (recall macro 0.99 en train, 0.74 en test).
- L2, Dropout et le modèle combiné réduisent fortement l'écart train/test et augmentent le recall, mais produisent plus de fausses alertes (F1 macro ≈ 0.59). BatchNorm et early stopping donnent le meilleur F1 macro (0.658).
- En maintenance, manquer une panne coûte plus cher qu'une fausse alerte : L2 ou le modèle combiné sont à privilégier.
- Test très petit (9 à 22 exemples par classe rare) : les petits écarts entre modèles ne sont pas concluants.

---

## Synthèse personnelle

-Construire un MLP PyTorch et à l'utiliser avec `GridSearchCV` via un estimateur compatible scikit-learn.
- Eviter éviter les fuites de données : découpage chronologique, standardisation sur le train .
- Un bon score ne suffit pas : en régression, il faut comparer à une référence naïve (ici elle bat mon réseau) ; sur données déséquilibrées, il faut regarder le F1 macro et le recall par classe plutôt que l'accuracy.
- La régularisation n'est utile que s'il y a du sur-apprentissage : elle a dégradé la régression mais amélioré la généralisation en classification.
- Avec très peu d'exemples par classe, il faut rester prudent sur les conclusions.
