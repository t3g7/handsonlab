 Hands on Machine Learning avec Apache Spark
 ===========================================

 Sujets couverts :
 -----------

 -  Brève intro sur le ML, les principaux algos et use cases
 -  Principes de base de Spark
 -  Spark ML Lib et Spark ML
 -  TP mise en oeuvre sur cas simple des deux librairies

 Brève intro
 -----------
 
 Le ML permet de construire des modèles prédictifs qui sauront retourner une information pour des données qu'ils ne connaissent pas.
 On trouve différents types de cas :
   - prédiction de valeurs (regression)
   - classification (binaire ou non)
   - moteurs de recommandation
   - clusterisation
 
 Dans certains cas, il s'agit d'algorithmes dits supervisés et dans d'autres, non supervisés.
 Un algorithme supervisé va dépendre d'informations qualifiées en entrée.
 Exemple : classification d'emails en spam et non spam
 Les algorithmes non supervisés n'auront pas besoin de cette information
 Exemple : le clustering 
 
 
 Principes de base de Spark
 --------------------------
 
 Spark permet d'opérer des calculs de façon distribuée sur des données en mémoire.
 
 Historiquement Spark travaille avec une représentation des données : les RDD (Resilient Distributed Datasets)
 Spark a depuis introduit une nouvelle notion : les DataFrames, notion existant aussi par ailleurs dans d'autres solutions (Panda)
  
 Les RDD sont une abstraction au dessus d'une collection de données.
 Les RDD vont pouvoir être créés depuis les données d'une source (file system, HDFS, base NoSQL, ...)
 Les données des RDD pourront être manipulées via des transformations (map, filter, ...)
 Chacune retournera un RDD : les RDD sont immutables
 Ces transformations sont lazy : elles ne sont appliquées qu'en fin de traitement...
 ... lorsqu'une action est appelée
 Une action ne retourne pas un RDD mais un résultat (ex. : count(), mean(), ...)
 
 Les DataFrames (Spark 1.3+) offrent une manipulation des données plus aisée via un DSL inspiré du SQL (select, where, groupBy...)
 Les DataFrames s'appuient sur un schéma associé aux données (implicitement ou explicitement défini)
 
 
 Spark et Scala
 
 Spark est écrit en Scala. Le langage permet une expression plus concise et sera préféré dans ce TP aux autres langages supportés (Java, Python)
 
 Notion de Tuple : une structure regroupant plusieurs valeurs de types différents potentiellement. Exemple : ("hello", 10)

 
 Spark MLLib et Spark ML
 -----------------------
 
 Historiquement Spark a fourni une librairie dédiée au Machine Learning : MLLib
 Récemment, le projet Spark ML est arrivé pour fournir une approche de plus haut niveau (comprendre : un peu moins technique et opaque) : Spark ML
 
 A ce jour, MLLib est toujours activement maintenue et étendue mais Spark ML est maintenant clairement mis en avant.
 
 MLLib va travailler avec les notions (types) de base suivantes :
  -  des vecteurs et des matrices, locaux ou distribués (RDD powered)
  -  un type dédié à l'apprentissage supervisé : le LabeledPoint (un vecteur accompagné d'un label) : le label sera un Double, permettant au LabeledVector de pouvoir servir tant pour la regression que pour la classification (binaire ou de classe multiple).
  
 MLLib va principalement travailler avec des représentations de type RDD
 
 Spark ML apporte une simplification ainsi que des facilités pour développer un traitement complet de type ML.
 Spark ML utilise la nouvelle représentation de données de Spark : les DataFrames (une sorte de RDD avec des labels sur chaque colonne)
 Spark ML introduit la notion de Pipeline pour modéliser la succession de transformations sur la data comme un tout.
 Spark ML propose aussi en plus des outils pour faciliter les phases de optimisation/sélection du modèle via les notions de Cross Validation ou TrainValidationSplit
 Spark ML est encore expérimental...
 
 TP
 --
 
 Probleme de clavier sous un hote Mac :
 
 Dans la VM : 
 sudo dpkg-reconfigure keyboard-configuration
 
 Choix du clavier Mac Book Pro, sauver, rebooter

 deux packages : *.mllib et *.ml dédiés chacun à une librairie
 
 dans chaque 2 classes : something et somethingAnswer... Ne travailler que sur something...
 
 dans chaque TP (mllib et ml), 2 parties :
 
   a) création et test de modèles
   
   b) optimisation et selection d'un modele