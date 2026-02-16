""
# Chapitre 3 : Apprentissage Supervisé (Supervised Learning)

L'apprentissage supervisé est le type d'apprentissage automatique le plus courant. Il consiste à entraîner un modèle sur un ensemble de données étiquetées, c'est-à-dire des données où la sortie souhaitée est déjà connue. L'objectif est que le modèle apprenne une fonction de mappage qui peut prédire la sortie pour de nouvelles données non vues.

Il existe deux principaux types de problèmes en apprentissage supervisé :

*   **Régression** : Prédire une valeur continue (par exemple, le prix d'une action, le montant d'un prêt).
*   **Classification** : Prédire une catégorie ou une classe discrète (par exemple, si un e-mail est un spam ou non, si une transaction est frauduleuse ou non).

#### 3.1 Régression Linéaire et Logistique

**Régression Linéaire**

La régression linéaire est l'un des algorithmes les plus simples. Elle tente de modéliser la relation entre une variable dépendante (la cible) et une ou plusieurs variables indépendantes (les caractéristiques) en ajustant une équation linéaire aux données.

L'équation d'une régression linéaire simple est :
$$ y = \beta_0 + \beta_1 x + \epsilon $$

Où :
*   $y$ est la variable dépendante.
*   $x$ est la variable indépendante.
*   $\beta_0$ est l'ordonnée à l'origine.
*   $\beta_1$ est le coefficient de la pente.
*   $\epsilon$ est le terme d'erreur.

**Régression Logistique**

Malgré son nom, la régression logistique est utilisée pour les problèmes de **classification binaire**. Elle prédit la probabilité qu'une observation appartienne à une classe donnée en utilisant la fonction sigmoïde (ou logistique).

La fonction sigmoïde transforme n'importe quelle valeur réelle en une valeur comprise entre 0 et 1 :
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Où $z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$. La sortie est la probabilité $P(y=1|x)$.

**Application en Finance : Scoring de Crédit**
La régression logistique est largement utilisée pour le scoring de crédit. Le modèle peut être entraîné sur des données historiques de prêts, où les caractéristiques sont les informations sur l'emprunteur (revenu, âge, dettes, etc.) et l'étiquette est de savoir si l'emprunteur a remboursé son prêt ou non. Le modèle peut alors prédire la probabilité de défaut pour un nouvel emprunteur.

#### 3.2 Arbres de Décision et Forêts Aléatoires

**Arbres de Décision**

Un arbre de décision est un modèle non paramétrique qui peut être utilisé pour la régression et la classification. Il apprend des règles de décision simples à partir des caractéristiques des données pour prédire la valeur de la cible. L'arbre est construit en divisant récursivement les données en sous-ensembles de plus en plus homogènes.

**Forêts Aléatoires (Random Forests)**

Une forêt aléatoire est un ensemble d'arbres de décision. Elle combine les prédictions de plusieurs arbres pour améliorer la précision et contrôler le surapprentissage (overfitting). Chaque arbre est entraîné sur un sous-ensemble aléatoire des données et des caractéristiques.

#### 3.3 Support Vector Machines (SVM)

Les Support Vector Machines (SVM) sont des modèles de classification puissants qui trouvent l'hyperplan optimal qui sépare les données en classes avec la plus grande marge possible. Les SVM peuvent également gérer des données non linéairement séparables en utilisant l'astuce du noyau (kernel trick).

#### 3.4 Exemple de Code Python : Prédiction de Défaut de Paiement

Voici un exemple simplifié de l'utilisation de la régression logistique pour prédire le défaut de paiement avec la bibliothèque `scikit-learn`.

```python
# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Chargement d'un jeu de données fictif
# En pratique, on utiliserait de vraies données de crédit
data = {'revenu': [50000, 80000, 30000, 120000, 45000],
        'dette': [20000, 10000, 15000, 5000, 25000],
        'defaut': [0, 0, 1, 0, 1]} # 0: non-défaut, 1: défaut
df = pd.DataFrame(data)

# Séparation des caractéristiques (X) et de la cible (y)
X = df[['revenu', 'dette']]
y = df['defaut']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement du modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Précision du modèle : {accuracy:.2f}")
print("Matrice de confusion :\n", conf_matrix)
```

#### 3.5 Synthèse et Auto-évaluation

**Synthèse du Chapitre 3**

Ce chapitre a introduit l'apprentissage supervisé, en se concentrant sur les tâches de régression et de classification. Nous avons exploré plusieurs algorithmes fondamentaux, de la régression linéaire et logistique aux arbres de décision et SVM. L'exemple de code a illustré comment appliquer ces concepts à un problème financier concret, le scoring de crédit.

**Questions d'Auto-évaluation**

1.  Quelle est la principale différence entre un problème de régression et un problème de classification ?
2.  Expliquez comment la régression logistique peut être utilisée pour un problème de classification.
3.  Quel est l'avantage d'utiliser une forêt aléatoire par rapport à un seul arbre de décision ?
4.  Dans l'exemple de code, que représente la matrice de confusion ?

---

**Références du Chapitre 3**

[1] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
[2] Müller, A. C., & Guido, S. (2016). *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media.
[3] Siddiqi, N. (2017). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
""
