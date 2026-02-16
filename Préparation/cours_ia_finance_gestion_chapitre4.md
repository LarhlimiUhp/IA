# Chapitre 4 : Apprentissage Non Supervisé (Unsupervised Learning)

Contrairement à l'apprentissage supervisé, l'apprentissage non supervisé traite des données non étiquetées. Son objectif est de découvrir des structures cachées, des motifs ou des relations au sein des données sans aucune connaissance préalable des résultats. Il est particulièrement utile pour l'exploration de données, la segmentation et la réduction de dimensionnalité.

#### 4.1 Clustering (K-means, DBSCAN) pour la Segmentation Client

Le **clustering** est une technique d'apprentissage non supervisé qui regroupe des points de données similaires en clusters. Les objets au sein d'un même cluster sont plus similaires entre eux qu'avec ceux d'autres clusters. En finance et gestion, le clustering est fréquemment utilisé pour la segmentation client, la détection d'anomalies et l'analyse de marché.

**K-means**

L'algorithme K-means est l'une des méthodes de clustering les plus populaires. Il vise à partitionner $n$ observations en $k$ clusters, où chaque observation appartient au cluster avec le centroïde (moyenne) le plus proche. L'algorithme est itératif et procède comme suit :

1.  Initialisation : Choisir $k$ centroïdes aléatoirement parmi les points de données.
2.  Affectation : Affecter chaque point de données au centroïde le plus proche.
3.  Mise à jour : Recalculer les centroïdes comme la moyenne des points de données affectés à chaque cluster.
4.  Répétition : Répéter les étapes 2 et 3 jusqu'à ce que les centroïdes ne changent plus significativement ou qu'un nombre maximal d'itérations soit atteint.

La fonction objectif que K-means cherche à minimiser est la somme des carrés des distances entre chaque point et son centroïde :
$$ J = \sum_{j=1}^{k} \sum_{i=1}^{n} \|x_i^{(j)} - c_j\|^2 $$
Où $x_i^{(j)}$ est un point de données appartenant au cluster $j$, et $c_j$ est le centroïde du cluster $j$.

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

DBSCAN est un algorithme de clustering basé sur la densité, capable de découvrir des clusters de formes arbitraires et de détecter le bruit (outliers). Il ne nécessite pas de spécifier le nombre de clusters à l'avance. DBSCAN définit les clusters comme des zones de haute densité séparées par des zones de faible densité.

Les points sont classifiés en trois types :
*   **Points centraux (Core points)** : Points ayant au moins `min_samples` points (y compris lui-même) dans un rayon `eps`.
*   **Points de frontière (Border points)** : Points qui sont à portée `eps` d'un point central, mais qui ne sont pas eux-mêmes des points centraux.
*   **Points de bruit (Noise points)** : Points qui ne sont ni des points centraux ni des points de frontière.

#### 4.2 Réduction de Dimensionnalité (PCA, t-SNE)

La **réduction de dimensionnalité** est le processus de réduction du nombre de variables aléatoires considérées en obtenant un ensemble de variables principales. Cela est crucial lorsque l'on travaille avec des données de haute dimension, car cela peut aider à visualiser les données, à réduire le bruit et à améliorer les performances des algorithmes d'apprentissage automatique.

**Analyse en Composantes Principales (PCA - Principal Component Analysis)**

La PCA est une technique linéaire de réduction de dimensionnalité qui transforme les données en un nouveau système de coordonnées, où les nouvelles dimensions (composantes principales) sont orthogonales et capturent la variance maximale des données. La première composante principale capture la plus grande variance, la deuxième la deuxième plus grande, et ainsi de suite.

Les étapes clés de la PCA incluent :
1.  Standardisation des données.
2.  Calcul de la matrice de covariance.
3.  Calcul des valeurs propres et vecteurs propres de la matrice de covariance.
4.  Sélection des $k$ vecteurs propres correspondant aux $k$ plus grandes valeurs propres pour former une matrice de projection.
5.  Projection des données originales sur ces nouvelles dimensions.

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**

t-SNE est une technique non linéaire de réduction de dimensionnalité particulièrement bien adaptée à la visualisation de données de haute dimension dans un espace de deux ou trois dimensions. Elle cherche à préserver la structure locale des données, c'est-à-dire que les points qui sont proches dans l'espace de haute dimension restent proches dans l'espace de faible dimension.

#### 4.3 Exemple de Code Python : Segmentation de Portefeuille Clients

Cet exemple utilise K-means pour segmenter un portefeuille clients fictif basé sur le revenu et la fréquence d'achat.

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Données clients fictives (revenu annuel en K€, fréquence d'achat par an)
data = {
    'ClientID': range(1, 21),
    'Revenu_K€': [30, 45, 70, 25, 80, 35, 60, 90, 20, 50, 75, 40, 65, 95, 15, 55, 85, 28, 72, 48],
    'Frequence_Achat': [5, 8, 12, 4, 15, 6, 10, 18, 3, 9, 13, 7, 11, 19, 2, 10, 16, 5, 14, 9]
}
df = pd.DataFrame(data)

# Sélection des caractéristiques pour le clustering
X = df[['Revenu_K€', 'Frequence_Achat']]

# Standardisation des données (très important pour K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-means
# On choisit k=3 pour cet exemple, mais en pratique on utiliserait la méthode du coude ou le score de silhouette
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init pour éviter les problèmes d'initialisation
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Revenu_K€', y='Frequence_Achat', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8)
plt.title('Segmentation Client par K-means')
plt.xlabel('Revenu Annuel (K€)')
plt.ylabel('Fréquence d\'Achat Annuelle')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
# plt.savefig('segmentation_clients.png') # Pour sauvegarder la figure
plt.show()

print(df.head())
print("\nCentroïdes des clusters (non-scalés, pour interprétation) :")
# Pour interpréter les centroïdes, il faut les 
dé-scaler.
print(scaler.inverse_transform(kmeans.cluster_centers_))

#### 4.4 Synthèse et Auto-évaluation

**Synthèse du Chapitre 4**

Ce chapitre a exploré l'apprentissage non supervisé, une approche puissante pour découvrir des structures cachées dans des données non étiquetées. Nous avons détaillé les techniques de clustering, comme K-means et DBSCAN, essentielles pour la segmentation client et la détection d'anomalies. La réduction de dimensionnalité, avec la PCA et t-SNE, a été présentée comme un moyen de simplifier les données complexes pour la visualisation et l'amélioration des modèles. L'exemple de code Python a illustré l'application pratique du clustering pour la segmentation de portefeuille clients.

**Questions d'Auto-évaluation**

1.  Quelle est la différence fondamentale entre l'apprentissage supervisé et non supervisé ?
2.  Expliquez le principe de l'algorithme K-means et ses étapes principales.
3.  Dans quel contexte la réduction de dimensionnalité est-elle utile ? Citez deux techniques.
4.  Comment le clustering peut-il être appliqué dans le domaine de la finance ou de la gestion ? Donnez un exemple concret.

---

**Références du Chapitre 4**

[1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.
[2] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
[3] MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability*, 1(14), 281-297.
[4] Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96 Proceedings*, 96(34), 226-231.
[5] Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
[6] Van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9(Nov), 2579-26056.
