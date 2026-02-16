# Chapitre 5 : Pré-processing et Sélection de Modèles

La qualité des données est primordiale pour la performance de tout modèle d'apprentissage automatique. Le pré-traitement des données est une étape cruciale qui vise à transformer les données brutes en un format propre et adapté aux algorithmes. Une fois les données préparées, la sélection et l'évaluation rigoureuse des modèles sont essentielles pour garantir leur robustesse et leur généralisabilité.

#### 5.1 Nettoyage et Normalisation des Données

Le pré-traitement des données englobe plusieurs techniques pour améliorer la qualité des données d'entrée.

*   **Nettoyage des Données** : Cette étape vise à gérer les incohérences, les erreurs et les valeurs aberrantes dans les données. Cela peut inclure la correction des fautes de frappe, la standardisation des formats, et la suppression ou la correction des enregistrements dupliqués.

*   **Traitement des Valeurs Manquantes** : Les valeurs manquantes sont courantes dans les ensembles de données réels et peuvent affecter la performance des modèles. Plusieurs stratégies existent pour les gérer :
    *   **Suppression** : Supprimer les lignes ou les colonnes contenant des valeurs manquantes. Cette méthode est simple mais peut entraîner une perte significative d'informations.
    *   **Imputation** : Remplacer les valeurs manquantes par des valeurs estimées. Les méthodes courantes incluent l'imputation par la moyenne, la médiane, le mode, ou des méthodes plus sophistiquées comme l'imputation par régression ou par k-plus proches voisins (k-NN).

*   **Normalisation et Mise à l'Échelle (Scaling)** : De nombreux algorithmes d'apprentissage automatique sont sensibles à l'échelle des caractéristiques. La normalisation ou la mise à l'échelle permet de ramener toutes les caractéristiques à une plage de valeurs similaire.
    *   **Standardisation (Z-score normalization)** : Transforme les données de sorte qu'elles aient une moyenne de 0 et un écart-type de 1.
        $$ x' = \frac{x - \mu}{\sigma} $$
        Où $\mu$ est la moyenne et $\sigma$ est l'écart-type de la caractéristique.
    *   **Min-Max Scaling** : Transforme les données pour qu'elles se situent dans une plage spécifique, généralement entre 0 et 1.
        $$ x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} $$
        Où $x_{\min}$ est la valeur minimale et $x_{\max}$ est la valeur maximale de la caractéristique.

#### 5.2 Métriques de Performance

L'évaluation des modèles est cruciale pour comprendre leur efficacité. Le choix des métriques dépend du type de problème (classification ou régression) et des objectifs spécifiques de l'application.

**Pour les problèmes de Classification** :

Considérons une classification binaire avec les termes suivants :
*   **Vrais Positifs (VP)** : Le modèle a correctement prédit la classe positive.
*   **Vrais Négatifs (VN)** : Le modèle a correctement prédit la classe négative.
*   **Faux Positifs (FP)** : Le modèle a prédit la classe positive alors qu'elle était négative (erreur de type I).
*   **Faux Négatifs (FN)** : Le modèle a prédit la classe négative alors qu'elle était positive (erreur de type II).

*   **Précision (Accuracy)** : Proportion des prédictions correctes parmi toutes les prédictions.
    $$ \text{Accuracy} = \frac{VP + VN}{VP + VN + FP + FN} $$

*   **Rappel (Recall ou Sensibilité)** : Proportion des vrais positifs correctement identifiés parmi toutes les observations positives réelles. Important lorsque le coût des faux négatifs est élevé (ex: détection de fraude).
    $$ \text{Recall} = \frac{VP}{VP + FN} $$

*   **Spécificité** : Proportion des vrais négatifs correctement identifiés parmi toutes les observations négatives réelles.
    $$ \text{Specificity} = \frac{VN}{VN + FP} $$

*   **Précision (Precision)** : Proportion des vrais positifs parmi toutes les observations prédites comme positives. Important lorsque le coût des faux positifs est élevé.
    $$ \text{Precision} = \frac{VP}{VP + FP} $$

*   **F1-Score** : Moyenne harmonique de la précision et du rappel, utile lorsque l'on recherche un équilibre entre les deux.
    $$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

*   **Courbe ROC (Receiver Operating Characteristic) et AUC (Area Under the Curve)** : La courbe ROC trace le taux de vrais positifs (Recall) en fonction du taux de faux positifs (1 - Spécificité) pour différents seuils de classification. L'AUC mesure l'aire sous cette courbe ; une AUC proche de 1 indique un excellent classifieur.

**Pour les problèmes de Régression** :

*   **Erreur Quadratique Moyenne (Mean Squared Error - MSE)** : Moyenne des carrés des différences entre les valeurs prédites et les valeurs réelles.
    $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

*   **Racine Carrée de l'Erreur Quadratique Moyenne (Root Mean Squared Error - RMSE)** : La racine carrée du MSE, plus interprétable car dans la même unité que la variable cible.
    $$ RMSE = \sqrt{MSE} $$

*   **Erreur Absolue Moyenne (Mean Absolute Error - MAE)** : Moyenne des valeurs absolues des différences entre les valeurs prédites et les valeurs réelles.
    $$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

#### 5.3 Validation Croisée et Optimisation des Hyperparamètres

*   **Validation Croisée (Cross-Validation)** : Technique utilisée pour évaluer la performance d'un modèle de manière plus robuste et éviter le surapprentissage. La méthode la plus courante est la **k-fold cross-validation** : les données sont divisées en $k$ sous-ensembles (folds). Le modèle est entraîné $k$ fois, chaque fois sur $k-1$ folds et évalué sur le fold restant. La performance finale est la moyenne des $k$ évaluations.

*   **Optimisation des Hyperparamètres** : Les hyperparamètres sont des paramètres qui ne sont pas appris par le modèle lui-même mais qui sont définis avant l'entraînement (ex: le nombre de voisins dans k-NN, le taux d'apprentissage dans la descente de gradient). L'optimisation des hyperparamètres vise à trouver la meilleure combinaison d'hyperparamètres pour un modèle donné.
    *   **Recherche par grille (Grid Search)** : Teste toutes les combinaisons possibles d'hyperparamètres définies dans une grille.
    *   **Recherche aléatoire (Random Search)** : Échantillonne aléatoirement des combinaisons d'hyperparamètres dans un espace défini.

#### 5.4 Exemple de Code Python : Pré-traitement et Évaluation de Modèle

Cet exemple montre comment appliquer le pré-traitement et évaluer un modèle de classification.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Création d'un jeu de données fictif avec des valeurs manquantes et des échelles différentes
data = {
    'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None, 110],
    'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Séparation des caractéristiques (X) et de la cible (y)
X = df[['feature1', 'feature2']]
y = df['target']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création d'un pipeline de pré-traitement et de modélisation
# 1. Imputation des valeurs manquantes par la moyenne
# 2. Standardisation des caractéristiques
# 3. Modèle de régression logistique
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Entraînement du pipeline
pipeline.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1] # Probabilités pour la classe positive

# Évaluation du modèle
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")

# Validation croisée
scores_cv = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
print(f"\nScores de validation croisée (Accuracy): {scores_cv}")
print(f"Accuracy moyenne CV: {scores_cv.mean():.2f}")

# Optimisation des hyperparamètres avec GridSearchCV
# Définition de la grille d'hyperparamètres à tester
param_grid = {
    'imputer__strategy': ['mean', 'median'],
    'scaler': [StandardScaler(), MinMaxScaler()], # Tester différents scalers
    'classifier__C': [0.1, 1.0, 10.0] # Paramètre de régularisation pour LogisticRegression
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"\nMeilleurs hyperparamètres: {grid_search.best_params_}")
print(f"Meilleur score de validation croisée: {grid_search.best_score_:.2f}")

# Utilisation du meilleur modèle trouvé
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Accuracy du meilleur modèle sur l'ensemble de test: {accuracy_score(y_test, y_pred_best):.2f}")
```

#### 5.5 Synthèse et Auto-évaluation

**Synthèse du Chapitre 5**

Ce chapitre a souligné l'importance cruciale du pré-traitement des données pour la construction de modèles d'IA performants. Nous avons abordé les techniques de nettoyage, de gestion des valeurs manquantes, et de normalisation. Ensuite, nous avons détaillé les métriques d'évaluation essentielles pour les problèmes de classification et de régression, ainsi que les méthodes de validation croisée et d'optimisation des hyperparamètres pour garantir la robustesse et la généralisabilité des modèles. L'exemple Python a illustré l'application pratique de ces concepts via un pipeline.

**Questions d'Auto-évaluation**

1.  Pourquoi le pré-traitement des données est-il une étape indispensable en apprentissage automatique ?
2.  Décrivez deux méthodes pour gérer les valeurs manquantes et expliquez quand utiliser l'une plutôt que l'autre.
3.  Quelle est la différence entre la Précision (Precision) et le Rappel (Recall) ? Dans quel cas l'une est-elle plus importante que l'autre ?
4.  Expliquez le principe de la validation croisée et son utilité.
5.  Qu'est-ce qu'un hyperparamètre et comment peut-on l'optimiser ?

---

**Références du Chapitre 5**

[1] Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.
[2] Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
[3] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.
[4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
