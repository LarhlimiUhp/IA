# Chapitre 6 : Réseaux de Neurones et Deep Learning

Le Deep Learning, ou apprentissage profond, est un sous-domaine de l'apprentissage automatique qui a révolutionné de nombreux domaines de l'intelligence artificielle, notamment la vision par ordinateur et le traitement du langage naturel. Il est basé sur des réseaux de neurones artificiels profonds, inspirés par la structure et le fonctionnement du cerveau humain.

#### 6.1 Le Neurone Artificiel et Fonctions d'Activation

Le bloc de construction fondamental des réseaux de neurones est le **neurone artificiel**, également appelé perceptron. Un neurone reçoit plusieurs entrées, les pondère, les somme, puis applique une fonction d'activation pour produire une sortie.

*   **Entrées ($x_i$)** : Les données d'entrée du neurone.
*   **Poids ($w_i$)** : Chaque entrée est multipliée par un poids, qui représente l'importance de cette entrée.
*   **Biais ($b$)** : Une valeur constante ajoutée à la somme pondérée, permettant au modèle de s'adapter à des données qui ne passent pas par l'origine.
*   **Somme pondérée ($z$)** : La somme des produits des entrées par leurs poids, plus le biais.
    $$ z = \sum_{i=1}^{n} (x_i w_i) + b $$
*   **Fonction d'activation ($\phi$)** : Une fonction non linéaire appliquée à la somme pondérée. Elle introduit la non-linéarité dans le modèle, permettant aux réseaux de neurones d'apprendre des relations complexes dans les données. Des exemples courants incluent :
    *   **Sigmoïde** : $\phi(z) = \frac{1}{1 + e^{-z}}$ (utilisée dans les couches de sortie pour la classification binaire).
    *   **ReLU (Rectified Linear Unit)** : $\phi(z) = \max(0, z)$ (très populaire dans les couches cachées).
    *   **Softmax** : Utilisée dans les couches de sortie pour la classification multi-classes, elle transforme les sorties en probabilités dont la somme est 1.

La sortie du neurone est alors $a = \phi(z)$.

#### 6.2 Perceptron Multicouche (MLP)

Un **Perceptron Multicouche (MLP)** est un réseau de neurones composé de plusieurs couches de neurones artificiels. Il comprend au moins trois couches :

1.  **Couche d'entrée** : Reçoit les données brutes.
2.  **Une ou plusieurs couches cachées** : Effectuent des transformations non linéaires sur les données.
3.  **Couche de sortie** : Produit la prédiction finale.

Les MLP sont entraînés en utilisant un algorithme appelé **rétropropagation du gradient (backpropagation)**, qui est une application de la descente de gradient pour ajuster les poids et les biais du réseau en fonction de l'erreur de prédiction.

#### 6.3 Introduction aux CNN et RNN

**Réseaux Neuronaux Convolutifs (CNN - Convolutional Neural Networks)**

Les CNN sont particulièrement efficaces pour le traitement des données ayant une structure de grille, comme les images. Ils utilisent des **couches de convolution** qui appliquent des filtres (noyaux) aux données d'entrée pour détecter des motifs locaux (bords, textures, formes). Les CNN sont largement utilisés en vision par ordinateur pour la reconnaissance d'images, la détection d'objets, etc.

**Réseaux Neuronaux Récurrents (RNN - Recurrent Neural Networks)**

Les RNN sont conçus pour traiter des séquences de données, où l'ordre des éléments est important. Contrairement aux MLP, les RNN ont des connexions qui forment des boucles, permettant aux informations de persister d'une étape à l'autre. Cela les rend idéaux pour le traitement du langage naturel (NLP), la reconnaissance vocale et l'analyse de séries temporelles. Des variantes plus avancées comme les **LSTM (Long Short-Term Memory)** et les **GRU (Gated Recurrent Unit)** ont été développées pour surmonter les problèmes de mémorisation à long terme des RNN simples.

#### 6.4 Exemple de Code Python : Prédiction de Séries Temporelles Financières avec LSTM

Cet exemple utilise un réseau LSTM pour prédire les valeurs futures d'une série temporelle financière (par exemple, le prix d'une action). Nous allons simuler des données pour illustrer le concept.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Génération de données de séries temporelles fictives (prix d'une action)
def generate_time_series(num_points):
    time = np.arange(0, num_points)
    amplitude = np.sin(time * 0.1) + np.random.uniform(-0.1, 0.1, num_points)
    return amplitude

series = generate_time_series(100)

# 2. Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1))

# 3. Préparation des données pour LSTM (création de séquences)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10 # Longueur de la séquence d'entrée pour la prédiction
X, y = create_sequences(series_scaled, seq_length)

# 4. Division en ensembles d'entraînement et de test
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 5. Construction du modèle LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer=\'adam\', loss=\'mean_squared_error\')

# 6. Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=0)

# 7. Prédiction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 8. Inverse la normalisation pour obtenir les valeurs réelles
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# 9. Visualisation des résultats
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(series_scaled), label=\'Données Réelles\')

# Ajuster les indices pour les prédictions d'entraînement
train_plot = np.empty_like(series_scaled)
train_plot[:, :] = np.nan
train_plot[seq_length:len(train_predict)+seq_length, :] = train_predict
plt.plot(train_plot, label=\'Prédictions Entraînement\')

# Ajuster les indices pour les prédictions de test
test_plot = np.empty_like(series_scaled)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+seq_length:len(series_scaled), :] = test_predict
plt.plot(test_plot, label=\'Prédictions Test\')

plt.title(\'Prédiction de Séries Temporelles Financières avec LSTM\')
plt.xlabel(\'Temps\')
plt.ylabel(\'Valeur\')
plt.legend()
plt.grid(True)
plt.show()
```

#### 6.5 Synthèse et Auto-évaluation

**Synthèse du Chapitre 6**

Ce chapitre a introduit le monde des réseaux de neurones et du Deep Learning. Nous avons commencé par les fondements du neurone artificiel et des fonctions d'activation, puis avons exploré l'architecture des Perceptrons Multicouches. Une attention particulière a été portée aux Réseaux Neuronaux Convolutifs (CNN) pour les données spatiales et aux Réseaux Neuronaux Récurrents (RNN), en particulier les LSTM, pour les données séquentielles. L'exemple de code Python a démontré l'application des LSTM à la prédiction de séries temporelles financières, illustrant la puissance du Deep Learning dans ce domaine.

**Questions d'Auto-évaluation**

1.  Décrivez les composants principaux d'un neurone artificiel et leur rôle.
2.  Pourquoi les fonctions d'activation non linéaires sont-elles cruciales dans les réseaux de neurones ?
3.  Quelle est la différence fondamentale entre un MLP, un CNN et un RNN ? Donnez un cas d'usage pour chacun.
4.  Expliquez pourquoi les LSTM sont souvent préférés aux RNN simples pour la prédiction de séries temporelles.

---

**Références du Chapitre 6**

[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
[2] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
[3] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
[4] Chollet, F. (2017). *Deep Learning with Python*. Manning Publications Co.
