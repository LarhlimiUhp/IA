# COURS COMPLET : INTELLIGENCE ARTIFICIELLE
## Finance, Gestion, Marketing, Commerce International & Supply Chain Management

**ENCG Settat - Niveau Master - 4ème année**  
**Année 2025-2026**

---

## Présentation du Manuel

Ce manuel universitaire constitue un support de cours exhaustif sur l'Intelligence Artificielle appliquée aux domaines de la finance, de la gestion, du marketing, du commerce international et du supply chain management. Il est conçu pour être autonome, ne nécessitant pas de présentation orale complémentaire. Chaque concept, formule mathématique et algorithme est expliqué en profondeur avec des exemples métiers concrets et des cas d'entreprises réels.

### Objectifs Pédagogiques

À l'issue de ce cours, vous serez capable de :

1. **Comprendre** les concepts fondamentaux de l'IA, du Machine Learning et du Deep Learning
2. **Maîtriser** les méthodes clés d'apprentissage supervisé, non supervisé et de deep learning
3. **Appliquer** l'IA à des problématiques concrètes de finance et de gestion
4. **Développer** des solutions reproductibles en Python avec pandas, scikit-learn et Keras
5. **Évaluer** les enjeux éthiques, les biais et la conformité (RGPD, AI Act)
6. **Piloter** des projets IA et dialoguer efficacement avec les équipes data/IT

### Prérequis

- **Mathématiques** : Statistiques descriptives et inférentielles, algèbre linéaire de base
- **Programmation** : Notions de base en Python
- **Métier** : Compréhension des KPI en finance et gestion, familiarité avec Excel/BI

### Modalités d'Évaluation

- **40%** : Contrôle continu (QCM, mini-projets, participation)
- **30%** : Projet pratique (étude de cas, code, rapport, soutenance)
- **30%** : Examen final (théorie + études de cas)

---

## Table des Matières

1. [Chapitre 1 — Introduction à l'Intelligence Artificielle](#chapitre-1)
2. [Chapitre 2 — Fondements Mathématiques](#chapitre-2)
3. [Chapitre 3 — Apprentissage Supervisé](#chapitre-3)
4. [Chapitre 4 — Apprentissage Non Supervisé](#chapitre-4)
5. [Chapitre 5 — Deep Learning](#chapitre-5)
6. [Chapitre 6 — Natural Language Processing (NLP)](#chapitre-6)
7. [Chapitre 7 — IA en Marketing](#chapitre-7)
8. [Chapitre 8 — IA en Commerce International](#chapitre-8)
9. [Chapitre 9 — IA en Supply Chain Management](#chapitre-9)
10. [Chapitre 10 — MLOps et Déploiement](#chapitre-10)

---

<a name="chapitre-1"></a>
# Chapitre 1 — Introduction à l'Intelligence Artificielle

## 1.1 Qu'est-ce que l'Intelligence Artificielle ?

### 1.1.1 Définition Formelle

L'Intelligence Artificielle (IA) est une branche de l'informatique qui vise à créer des systèmes capables d'effectuer des tâches qui nécessitent traditionnellement l'intelligence humaine. Ces tâches incluent :

- Le **raisonnement** et la résolution de problèmes complexes
- La **perception** et l'interprétation de données sensorielles
- La **compréhension** du langage naturel
- La **prise de décision** dans des environnements incertains
- L'**apprentissage** à partir de l'expérience

Formellement, l'IA cherche à modéliser et à implémenter des fonctions cognitives en utilisant des algorithmes et des architectures computationnelles.

### 1.1.2 Définition Intuitive

Imaginez un système informatique qui peut :
- **Apprendre** de ses erreurs comme un étudiant qui s'améliore après chaque examen
- **S'adapter** à de nouvelles situations sans être reprogrammé pour chaque cas
- **Reconnaître** des patterns complexes que l'œil humain pourrait manquer
- **Prédire** des événements futurs en analysant des données historiques

L'IA, c'est essentiellement donner aux machines la capacité d'imiter l'intelligence humaine, non pas en copiant le cerveau, mais en utilisant des approches mathématiques et statistiques pour arriver à des résultats similaires.

### 1.1.3 Exemple Concret

**Reconnaissance faciale sur votre smartphone** :
- Lorsque vous déverrouillez votre téléphone avec votre visage, un système d'IA :
  1. **Détecte** votre visage dans l'image capturée par la caméra
  2. **Extrait** des caractéristiques uniques (distance entre les yeux, forme du nez, etc.)
  3. **Compare** ces caractéristiques avec le modèle stocké lors de l'enregistrement
  4. **Décide** de déverrouiller ou non en quelques millisecondes

Ce processus utilise des réseaux de neurones profonds entraînés sur des millions d'images de visages.

### 1.1.4 Exemples Métier

#### Finance : Détection de Fraude Bancaire
Les banques utilisent l'IA pour analyser en temps réel les transactions et détecter les comportements suspects :
- **Analyse de patterns** : Une transaction de 5000€ à 3h du matin au Japon alors que vous vivez au Maroc
- **Score de risque** : Attribution automatique d'un score de probabilité de fraude
- **Décision automatisée** : Blocage temporaire de la carte et envoi d'une alerte SMS
- **Apprentissage continu** : Le système s'améliore en analysant les retours (vraie fraude vs fausse alerte)

#### Gestion : Prédiction du Turnover RH
Les entreprises prédisent quels employés risquent de démissionner :
- **Variables analysées** : Ancienneté, salaire, promotions, satisfaction, absences
- **Modèle prédictif** : Calcul d'un score de risque de départ pour chaque employé
- **Action préventive** : Programmes de rétention ciblés sur les employés à risque
- **ROI mesurable** : Réduction de 20-30% du turnover dans certaines entreprises

### 1.1.5 Schéma Explicatif (Description Textuelle)

```
┌─────────────────────────────────────────────────────────────┐
│              SYSTÈME D'INTELLIGENCE ARTIFICIELLE            │
└─────────────────────────────────────────────────────────────┘
                               │
                               ↓
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   DONNÉES        │───>│   ALGORITHMES    │───>│   DÉCISIONS/     │
│   D'ENTRÉE       │    │   D'APPRENTISSAGE│    │   PRÉDICTIONS    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                        │
        │                       ↓                        │
        │              ┌──────────────────┐              │
        │              │    MODÈLE        │              │
        │              │    ENTRAÎNÉ      │              │
        │              └──────────────────┘              │
        │                       │                        │
        └───────────────────────┼────────────────────────┘
                                │
                                ↓
                        ┌──────────────────┐
                        │   FEEDBACK       │
                        │   ET AMÉLIORATION│
                        └──────────────────┘
```

**Explication du schéma** :
1. **Données d'entrée** : Informations brutes (images, textes, chiffres, transactions)
2. **Algorithmes d'apprentissage** : Méthodes mathématiques qui analysent les données
3. **Modèle entraîné** : Représentation mathématique des patterns découverts
4. **Décisions/Prédictions** : Résultats produits par le système
5. **Feedback** : Évaluation des résultats pour améliorer le modèle
6. **Boucle d'amélioration** : Le système s'améliore continuellement grâce au feedback

### 1.1.6 Cas Réel d'Entreprise : Netflix

**Contexte** :
Netflix compte plus de 230 millions d'abonnés dans le monde qui choisissent parmi un catalogue de plus de 15 000 titres. Sans IA, trouver un contenu intéressant serait comme chercher une aiguille dans une botte de foin.

**Solution IA Mise en Place** :
Netflix utilise un système de recommandation sophistiqué qui combine plusieurs algorithmes d'IA :

1. **Filtrage Collaboratif** :
   - Analyse les habitudes de visionnage de millions d'utilisateurs
   - Identifie des groupes d'utilisateurs avec des goûts similaires
   - Recommande ce que des utilisateurs similaires ont aimé

2. **Filtrage basé sur le Contenu** :
   - Analyse les métadonnées de chaque film/série (genre, acteurs, réalisateur, thèmes)
   - Identifie les caractéristiques des contenus que vous avez aimés
   - Suggère des titres avec des caractéristiques similaires

3. **Apprentissage Profond** :
   - Analyse les images des vignettes pour prédire lesquelles vous attireront
   - Personnalise même les vignettes affichées selon vos préférences
   - Optimise l'ordre de présentation des recommandations

**Résultats Mesurables** :
- **80%** du contenu visionné provient des recommandations de l'IA
- **Économie de 1 milliard de dollars** par an en rétention d'abonnés
- **Engagement accru** : Les utilisateurs passent moins de temps à chercher et plus à regarder
- **Personnalisation** : Chaque utilisateur voit une page d'accueil unique

**Technologies Utilisées** :
- Python, Apache Spark pour le traitement big data
- TensorFlow et PyTorch pour les modèles de deep learning
- A/B testing continu pour optimiser les algorithmes

---

## 1.2 Machine Learning (Apprentissage Automatique)

### 1.2.1 Définition Formelle

Le Machine Learning (ML) est une sous-discipline de l'IA qui se concentre sur le développement d'algorithmes permettant aux ordinateurs d'apprendre à partir de données et d'améliorer leurs performances sur une tâche spécifique sans être explicitement programmés pour chaque cas.

Formellement, un programme informatique apprend de l'expérience E par rapport à une classe de tâches T et une mesure de performance P, si sa performance sur T, mesurée par P, s'améliore avec l'expérience E.

**Équation fondamentale du ML** :
```
Apprendre : Données + Algorithme → Modèle
Prédire  : Nouvelles Données + Modèle → Prédictions
```

### 1.2.2 Définition Intuitive

Le Machine Learning, c'est comme enseigner à un enfant à reconnaître des animaux :
- Au lieu de programmer des règles explicites ("si ça a 4 pattes, un museau et aboie, c'est un chien")
- On montre des **exemples** (des milliers de photos de chiens)
- L'algorithme **découvre lui-même** les patterns qui caractérisent un chien
- Il peut ensuite **généraliser** pour reconnaître de nouveaux chiens jamais vus

La différence avec la programmation traditionnelle :
- **Programmation classique** : Règles + Données → Résultats
- **Machine Learning** : Données + Résultats → Règles (modèle)

### 1.2.3 Exemple Concret

**Filtre anti-spam d'email** :

**Approche traditionnelle (programmation classique)** :
```python
# Règles explicites programmées manuellement
if "viagra" in email or "lottery" in email or "prince" in email:
    return "SPAM"
else:
    return "HAM" (légitime)
```
Problème : Les spammeurs contournent facilement ces règles

**Approche Machine Learning** :
```python
# 1. Collecter des exemples
emails_spam = ["Buy viagra now!", "You won the lottery!", ...]
emails_ham = ["Meeting at 3pm", "Project report attached", ...]

# 2. Entraîner un modèle
model = NaiveBayes()
model.fit(emails, labels)  # Apprend automatiquement les patterns

# 3. Prédire sur de nouveaux emails
new_email = "Congratulations! Click here to claim prize"
prediction = model.predict(new_email)  # → SPAM (probabilité: 95%)
```

Le modèle apprend automatiquement que :
- Certains mots sont plus fréquents dans les spams ("free", "win", "click")
- La structure des phrases diffère (beaucoup de ponctuation, MAJUSCULES)
- Les spams contiennent souvent des liens suspects

### 1.2.4 Exemples Métier

#### Finance : Scoring de Crédit Automatisé

**Problématique** :
Une banque reçoit 10 000 demandes de crédit par mois. Comment décider rapidement et objectivement qui obtient un prêt ?

**Solution ML** :
```
1. DONNÉES HISTORIQUES (5 ans)
   - 100 000 dossiers de crédit passés
   - Pour chaque client : âge, revenu, emploi, dettes, historique bancaire
   - Résultat connu : a remboursé (0) ou fait défaut (1)

2. ENTRAÎNEMENT DU MODÈLE
   - Algorithme : Régression Logistique ou XGBoost
   - Le modèle apprend les combinaisons de facteurs qui prédisent le défaut
   - Exemples de patterns découverts :
     * Ratio dette/revenu > 40% → risque élevé
     * Emploi stable + épargne >50k → risque faible
     * Âge < 25 + crédit > 20k → risque moyen-élevé

3. MISE EN PRODUCTION
   - Nouveau dossier → Score de 0 à 1000
   - Score > 700 : Approbation automatique
   - Score 400-700 : Examen manuel
   - Score < 400 : Refus automatique

4. RÉSULTATS
   - Temps de décision : 48h → 5 minutes
   - Taux de défaut réduit de 15%
   - Objectivité accrue (réduction des biais humains)
```

#### Gestion : Prévision des Ventes pour Optimiser les Stocks

**Problématique** :
Un retailer doit commander les quantités optimales pour chaque magasin et chaque produit. Trop = invendus, Pas assez = ruptures = clients mécontents.

**Solution ML** :
```
1. DONNÉES COLLECTÉES
   - Historique des ventes (3 ans, par jour, produit, magasin)
   - Facteurs externes : météo, jours fériés, promotions, événements
   - Caractéristiques produit : catégorie, prix, saisonnalité

2. MODÈLE DE PRÉVISION
   - Algorithme : LSTM (réseaux de neurones récurrents) ou Prophet
   - Le modèle apprend :
     * Les tendances saisonnières (+ ventes de glaces en été)
     * L'impact des promotions (+30% lors des -20%)
     * Les corrélations entre produits (chips + sodas)

3. PRÉDICTIONS QUOTIDIENNES
   - Prévision à 7 jours : Quantité attendue par produit/magasin
   - Intervalle de confiance : Min-Max pour gérer l'incertitude
   - Alertes automatiques si risque de rupture

4. IMPACT BUSINESS
   - Réduction des ruptures de stock : -25%
   - Diminution des invendus : -18%
   - Satisfaction client : +12%
   - ROI : 300% la première année
```

### 1.2.5 Schéma Explicatif : Les Trois Types d'Apprentissage

```
┌────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING                            │
└────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ↓                  ↓                  ↓
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  APPRENTISSAGE  │  │  APPRENTISSAGE   │  │  APPRENTISSAGE   │
│    SUPERVISÉ    │  │   NON SUPERVISÉ  │  │  PAR RENFORCEMENT│
└─────────────────┘  └──────────────────┘  └──────────────────┘

Données étiquetées    Données non          Agent + Environnement
(X, y)               étiquetées (X)        Récompenses/Pénalités

Exemples:             Exemples:            Exemples:
• Classification      • Clustering         • Jeux (échecs, Go)
• Régression         • Réduction dimension • Robots autonomes
• Prédiction         • Détection anomalies • Trading algorithmique

Algorithmes:          Algorithmes:         Algorithmes:
• Régression linéaire • K-means           • Q-Learning
• Random Forest      • PCA               • Deep Q-Network
• Neural Networks    • DBSCAN            • AlphaGo
```

**Explication détaillée des trois types** :

1. **Apprentissage Supervisé** (comme un élève avec un professeur)
   - On fournit des **exemples avec les réponses** (données étiquetées)
   - Le modèle apprend la relation entre entrées (X) et sorties (y)
   - Utilisé pour : prédire, classifier, estimer

2. **Apprentissage Non Supervisé** (comme un explorateur)
   - On fournit des **données sans réponses** (non étiquetées)
   - Le modèle découvre la structure cachée dans les données
   - Utilisé pour : segmenter, réduire la dimensionnalité, détecter des anomalies

3. **Apprentissage par Renforcement** (comme un enfant qui apprend par essai-erreur)
   - Un **agent** interagit avec un **environnement**
   - Il reçoit des **récompenses** pour les bonnes actions, des **pénalités** pour les mauvaises
   - Il apprend la **stratégie optimale** pour maximiser les récompenses cumulées

### 1.2.6 Cas Réel d'Entreprise : Amazon

**Système de Recommandation "Customers Who Bought This Also Bought"**

**Contexte** :
Amazon vend des centaines de millions de produits. Comment suggérer les bons produits à chaque client parmi cette immensité ?

**Solution ML Déployée** :

**1. Collecte de Données Massives**
```
Pour chaque utilisateur :
- Historique d'achats
- Produits consultés
- Temps passé sur chaque page
- Produits ajoutés au panier puis retirés
- Recherches effectuées
- Avis laissés
```

**2. Algorithmes ML Utilisés**

a) **Filtrage Collaboratif (Item-to-Item)**
```python
# Pseudo-code simplifié
# Si vous achetez un livre de science-fiction:

similarities = {}
for other_item in all_items:
    users_who_bought_both = count_users(book, other_item)
    similarity_score = cosine_similarity(book, other_item)
    similarities[other_item] = similarity_score

# Recommander les items avec les scores les plus élevés
top_recommendations = sorted(similarities)[:10]
```

b) **Analyse des Patterns d'Achat**
```
Pattern découvert : 
Achat(Nintendo Switch) → forte probabilité d'acheter :
- Jeux Switch (95%)
- Manette supplémentaire (70%)
- Pochette de transport (60%)
- Carte SD (55%)

→ Afficher ces produits dans "Fréquemment achetés ensemble"
```

**3. Architecture Technique**
```
┌──────────────────┐
│  DONNÉES CLIENT  │ → Historique + Comportement temps réel
└────────┬─────────┘
         │
         ↓
┌──────────────────────────────────────┐
│  MODÈLES ML (plusieurs algorithmes)  │
├──────────────────────────────────────┤
│ • Collaborative Filtering            │
│ • Deep Learning (embeddings)         │
│ • Association Rules Mining           │
│ • Sequential Pattern Mining          │
└────────┬─────────────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│  PERSONNALISATION EN TEMPS   │
│  RÉEL                        │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│  RECOMMANDATIONS AFFICHÉES   │
│  (< 100ms de latence)        │
└──────────────────────────────┘
```

**4. Résultats Business**
- **35% du chiffre d'affaires** d'Amazon provient des recommandations
- **Augmentation de 29%** du panier moyen
- **Taux de conversion** : +15% pour les utilisateurs qui cliquent sur les recommandations
- **Valeur estimée** : Plusieurs milliards de dollars de revenus additionnels par an

**5. Améliorations Continues**
Amazon utilise l'A/B testing en permanence :
- Teste différentes versions d'algorithmes simultanément
- Mesure l'impact sur les KPIs (taux de clic, conversion, revenus)
- Déploie automatiquement la meilleure variante
- Réentraîne les modèles quotidiennement avec les nouvelles données

---

## 1.3 Deep Learning (Apprentissage Profond)

### 1.3.1 Définition Formelle

Le Deep Learning est une sous-catégorie du Machine Learning basée sur des réseaux de neurones artificiels comportant **plusieurs couches cachées** (d'où le terme "profond"). Ces architectures sont capables d'apprendre des représentations hiérarchiques des données, passant de caractéristiques simples (bords, textures) à des concepts complexes (objets, scènes).

Mathématiquement, un réseau de neurones profond est une composition de fonctions :

```
f(x) = f_L(f_{L-1}(...f_2(f_1(x; θ₁); θ₂)...; θ_{L-1}); θ_L)
```

Où :
- `x` = données d'entrée
- `f_i` = fonction de la couche i (transformation linéaire + activation non-linéaire)
- `θ_i` = paramètres (poids et biais) de la couche i
- `L` = nombre de couches

### 1.3.2 Définition Intuitive

Imaginez que vous voulez reconnaître un chat dans une image :

**Approche ML Classique** :
- Vous devez définir manuellement les caractéristiques pertinentes
- "Extrayez la forme des oreilles, comptez les moustaches, mesurez la taille des yeux..."
- C'est fastidieux et vous pourriez manquer des caractéristiques importantes

**Approche Deep Learning** :
- Vous montrez 100 000 images de chats au réseau
- **Couche 1** apprend automatiquement les bords et les coins
- **Couche 2** combine les bords pour détecter des textures (poils, motifs)
- **Couche 3** assemble les textures en parties (oreilles, yeux, museau)
- **Couche 4** combine les parties pour reconnaître "un chat"
- Tout cela **sans que vous ayez à programmer explicitement** ces étapes !

Analogie : C'est comme si un enfant apprenait à reconnaître des concepts de plus en plus abstraits :
- Niveau 1 : Lignes et formes
- Niveau 2 : Combinaisons de formes
- Niveau 3 : Parties d'objets
- Niveau 4 : Objets complets
- Niveau 5 : Scènes et contextes

### 1.3.3 Exemple Concret

**Reconnaissance d'écriture manuscrite (MNIST)**

Imaginons que vous voulez créer un système qui lit automatiquement les chiffres écrits à la main (0-9) :

**Données** :
- 60 000 images de chiffres manuscrits (28×28 pixels en niveaux de gris)
- Chaque pixel a une valeur de 0 (blanc) à 255 (noir)

**Architecture du Réseau de Neurones** :
```
INPUT → CONV1 → POOL1 → CONV2 → POOL2 → FC1 → FC2 → OUTPUT

784 pixels     128        64        32       128      10
(28×28)      neurones  neurones  neurones neurones classes
                                                    (0-9)
```

**Ce que chaque couche apprend** :
1. **CONV1** (Convolution 1) : Détecte des bords simples
   - Filtres qui reconnaissent lignes verticales, horizontales, diagonales
   
2. **POOL1** (Pooling 1) : Réduit la taille en gardant l'essentiel
   - Conserve les informations importantes, ignore les détails
   
3. **CONV2** (Convolution 2) : Détecte des formes plus complexes
   - Coins, courbes, boucles caractéristiques des chiffres
   
4. **POOL2** (Pooling 2) : Nouvelle réduction dimensionnelle
   
5. **FC1** (Fully Connected 1) : Combine les caractéristiques
   - Apprend les combinaisons qui distinguent les chiffres
   
6. **FC2** (Fully Connected 2) : Décision finale
   - Produit 10 probabilités (une par chiffre)
   - Ex : [0.01, 0.02, 0.85, 0.05, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01]
   - → Le chiffre est probablement un "2" (85% de confiance)

**Résultats** :
- Précision : 99.2% sur le jeu de test
- Temps d'inférence : < 1ms par image
- Applications : Lecture automatique de chèques, codes postaux, formulaires

### 1.3.4 Exemples Métier

#### Finance : Prédiction de Séries Temporelles Financières

**Problématique** :
Prédire les mouvements de prix d'actions en analysant des années de données historiques complexes (prix, volumes, indicateurs techniques, news, sentiment sur réseaux sociaux).

**Solution Deep Learning : LSTM (Long Short-Term Memory)**

```python
# Architecture LSTM pour prédire le prix de demain

Données d'entrée (séquence de 60 jours) :
┌──────────────────────────────────────────┐
│ Jour 1  Jour 2  ...  Jour 59  Jour 60   │
│ [Prix, Volume, RSI, MACD, Sentiment]    │
└──────────────────────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Couche LSTM (128)      │ ← Mémorise patterns à long terme
└──────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Couche LSTM (64)       │ ← Raffine les patterns
└──────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Couche Dense (32)      │ ← Combine les informations
└──────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Sortie (1)             │ ← Prix prédit pour demain
└──────────────────────────┘
```

**Ce que le modèle apprend** :
- **Patterns temporels** : "Après une séquence de 5 jours de hausse modérée, correction probable"
- **Corrélations complexes** : "Volume élevé + sentiment négatif → baisse imminente"
- **Saisonnalité** : "Historiquement, baisse en septembre (fiscal year-end)"
- **Réaction aux news** : "Annonce de bénéfices supérieurs → hausse dans les 3 jours suivants"

**Performances** :
- RMSE (erreur) : 2.3% sur données de test
- Direction correcte prédite : 62% du temps (vs 50% au hasard)
- **Attention** : Pas de "boule de cristal" ! Les marchés restent imprévisibles
- Usage : Aide à la décision, pas trading automatique sans supervision

**Mise en garde** :
Les marchés financiers sont influencés par des facteurs externes imprévisibles (crises, guerres, décisions politiques). Le DL améliore les prédictions mais ne garantit jamais la performance.

#### Gestion : Analyse de Sentiment Client à partir d'Avis

**Problématique** :
Une entreprise e-commerce reçoit 10 000 avis clients par jour. Comment identifier rapidement les problèmes et les opportunités d'amélioration ?

**Solution Deep Learning : Transformers (BERT fine-tuné)**

```
INPUT : Texte de l'avis client
"Le produit est arrivé rapidement mais la qualité est décevante. 
Le service client a été excellent pour gérer le retour."

         ↓
┌────────────────────────────────────┐
│  TOKENIZATION                      │
│  [Le, produit, est, arrivé, ...]   │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│  EMBEDDINGS (contextuels)          │
│  Chaque mot → vecteur de 768 dim   │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│  BERT ENCODER (12 couches)         │
│  Comprend contexte et nuances      │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│  CLASSIFICATION HEAD               │
└────────────────────────────────────┘
         ↓
OUTPUT :
• Sentiment global : MITIGÉ (score: 0.52)
• Aspects identifiés :
  - Livraison : POSITIF (0.89)
  - Qualité produit : NÉGATIF (0.15)
  - Service client : POSITIF (0.93)
```

**Ce que le modèle comprend** :
- **Nuances** : "rapidement" (positif) et "décevante" (négatif) dans même phrase
- **Aspects multiples** : Distingue qualité produit vs service client
- **Contexte** : "excellent" s'applique au service, pas au produit
- **Sarcasme** (limité) : Détecte certaines formes d'ironie

**Résultats Business** :
- **Automatisation** : Classification de 95% des avis sans intervention humaine
- **Alertes automatiques** : Email au chef de produit si >10 avis négatifs sur qualité en 24h
- **Dashboard temps réel** : Sentiment par catégorie de produit, évolution temporelle
- **ROI** : Économie de 50 000€/an en analyse manuelle + amélioration produits

### 1.3.5 Schéma Explicatif : Architectures Deep Learning par Type de Données

```
┌──────────────────────────────────────────────────────────────────┐
│                    DEEP LEARNING ARCHITECTURES                   │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│   IMAGES / VISION   │  │  SÉQUENCES / TEMPS   │  │    TEXTE / NLP      │
└─────────────────────┘  └──────────────────────┘  └─────────────────────┘
          │                        │                         │
          ↓                        ↓                         ↓
    CNN (Réseaux de          RNN / LSTM / GRU          Transformers
     Convolution)           (Réseaux Récurrents)        (Attention)
          │                        │                         │
          ↓                        ↓                         ↓
┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│ COUCHES CONV        │  │ CELLULES MÉMOIRE     │  │ SELF-ATTENTION      │
│ • Filtres détectent │  │ • Mémorisent passé   │  │ • Relations entre   │
│   patterns locaux   │  │ • Gèrent dépendances │  │   mots              │
│ • Hiérarchie        │  │   temporelles        │  │ • Parallélisation   │
│   caractéristiques  │  │                      │  │   efficace          │
└─────────────────────┘  └──────────────────────┘  └─────────────────────┘
          │                        │                         │
          ↓                        ↓                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                       APPLICATIONS                              │
├──────────────────┬────────────────────┬─────────────────────────┤
│ • Reconnaissance │ • Prédiction prix  │ • Traduction           │
│   faciale        │ • Détection fraude │ • Chatbots             │
│ • Diagnostic     │   temporelle       │ • Analyse sentiment    │
│   médical        │ • Prévision ventes │ • Résumé texte         │
│ • Conduite auto  │ • Trading algo     │ • Q&A                  │
└──────────────────┴────────────────────┴─────────────────────────┘
```

**Explication des architectures** :

1. **CNN (Convolutional Neural Networks)** - Pour les images
   - **Principe** : Filtres glissants qui détectent des patterns locaux
   - **Couches** : Convolution → Pooling → Convolution → Pooling → Dense
   - **Force** : Invariance à la translation (détecte un chat peu importe sa position)
   - **Exemple** : ResNet (152 couches), utilisé par Facebook pour taguer automatiquement les photos

2. **RNN/LSTM** - Pour les séquences temporelles
   - **Principe** : Cellules avec mémoire qui traitent les données séquentiellement
   - **LSTM** : Amélioration du RNN, gère mieux les dépendances à long terme
   - **Force** : Comprend le contexte temporel
   - **Exemple** : Google Translate (avant Transformers)

3. **Transformers** - Pour le texte et au-delà
   - **Principe** : Mécanisme d'attention pour capter les relations entre tous les mots
   - **Force** : Parallélisable (plus rapide), gère mieux les longues séquences
   - **Exemples** : BERT (Google), GPT (OpenAI), LLaMA (Meta)

### 1.3.6 Cas Réel d'Entreprise : Tesla - Autopilot

**Contexte** :
Tesla développe un système de conduite autonome qui doit comprendre l'environnement routier en temps réel à partir de 8 caméras montées sur le véhicule.

**Défi Technique** :
- Traiter 8 flux vidéo simultanément (1280×960 pixels, 36 FPS)
- Identifier : voitures, piétons, cyclistes, panneaux, marquages au sol, feux de signalisation
- Prédire les trajectoires des autres véhicules
- Prendre des décisions en moins de 100ms
- Fonctionner dans toutes les conditions (jour, nuit, pluie, neige)

**Architecture Deep Learning** :

```
┌──────────────────────────────────────────────────────────────┐
│              8 CAMÉRAS → FLUX VIDÉO EN TEMPS RÉEL            │
└──────────────────────────────────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────┐
│          RÉSEAU DE NEURONES CONVOLUTIONNEL (CNN)             │
│                      HydraNet Architecture                    │
├──────────────────────────────────────────────────────────────┤
│  BACKBONE (tronc commun) - ResNet modifié                    │
│  • 50 couches de convolution                                 │
│  • Extrait caractéristiques visuelles                        │
└─────────────┬────────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┬─────────────┬──────────────┐
    │         │         │             │              │
    ↓         ↓         ↓             ↓              ↓
┌────────┐ ┌──────┐ ┌─────────┐ ┌──────────┐ ┌─────────────┐
│DÉTECTION│ │SEGMEN│ │ ESTIMA- │ │PRÉDICTION│ │ DÉTECTION   │
│OBJETS   │ │TATION│ │ TION    │ │TRAJECTO  │ │ PROFONDEUR  │
│(Bounding│ │SÉMAN-│ │ PROFOND.│ │-IRES     │ │ (Distance)  │
│Boxes)   │ │TIQUE │ │         │ │          │ │             │
└────────┘ └──────┘ └─────────┘ └──────────┘ └─────────────┘
    │         │         │             │              │
    └─────────┴─────────┴─────────────┴──────────────┘
                        │
                        ↓
              ┌──────────────────────┐
              │  MODULE DE FUSION    │
              │  Combine toutes les  │
              │  informations        │
              └──────────┬───────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │  PLANIFICATION       │
              │  DE TRAJECTOIRE      │
              └──────────┬───────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │  COMMANDES VÉHICULE  │
              │  Accélération        │
              │  Freinage            │
              │  Direction           │
              └──────────────────────┘
```

**Données d'Entraînement** :
- **10 milliards de miles** parcourus par la flotte Tesla
- **1 milliard d'images** annotées (objets, distances, trajectoires)
- **Annotation semi-automatique** : Les modèles actuels aident à annoter de nouvelles données
- **Shadow Mode** : Les nouvelles versions tournent en arrière-plan pour collecter des données sans contrôler le véhicule

**Entraînement du Modèle** :
```
Matériel :
- 10 000 GPUs NVIDIA A100
- Supercalculateur Dojo (conçu par Tesla)
- 1 Exaflop de puissance de calcul

Temps d'entraînement :
- 2-3 semaines pour une version majeure
- Réentraînement continu avec nouvelles données

Optimisations :
- Quantification (réduction précision : FP32 → INT8)
- Pruning (suppression neurones inutiles)
- Distillation (transfert vers modèle plus petit)
→ Objectif : Tenir sur le hardware embarqué (chip FSD)
```

**Déploiement** :
```
Tesla FSD Chip (Hardware 3.0) :
- 144 TOPS (Trillions Operations Per Second)
- Consommation : 72 Watts
- Redondance : 2 chips indépendants (sécurité)
- Mise à jour OTA (Over-The-Air) du modèle

Pipeline d'inférence :
Images (8 caméras) → Preprocessing → Modèle CNN
                                       ↓
                                  Détections
                                       ↓
                                  Fusion sensorielle
                                       ↓
                                  Décisions
                                       ↓
                        Commandes véhicule (< 100ms)
```

**Résultats et Défis** :

✅ **Succès** :
- **10 fois moins d'accidents** qu'un conducteur humain moyen (selon Tesla)
- **Amélioration continue** : Chaque version (FSD 11, 12, 13...) plus performante
- **Généralisation** : Fonctionne dans des scenarios jamais vus en entraînement

⚠ **Défis Restants** :
- **Edge cases** : Situations rares (cône de chantier inhabituel, main d'un policier)
- **Conditions extrêmes** : Neige épaisse, brouillard dense
- **Responsabilité légale** : Qui est responsable en cas d'accident ?
- **Biais géographiques** : Performant aux USA, moins en Europe/Asie

**Enseignements pour l'IA en Entreprise** :
1. **Données = Carburant** : Plus de données → meilleur modèle
2. **Itération continue** : Déployer, collecter feedback, améliorer
3. **Infrastructu