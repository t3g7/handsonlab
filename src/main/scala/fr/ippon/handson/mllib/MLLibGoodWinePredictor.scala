package fr.ippon.handson.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.rdd.RDD

/**
 * @author dmartin
 */
object MLLibGoodWinePredictor {

  // Cette valeur sert à classer un bon vin (note >=6) d'un vin... moins bon (<6)
  val qualityLevel = 6

  // Creation de la configuration de Spark
  val conf = new SparkConf()
    .setAppName("ML with Spark (Hands-on) - Classification with MLLib")
    .setMaster("local[*]")

  // Creation du contexte Spark
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {

    println("Classification avec MLLib")

    printCitation()

    // Commencer par executer le job Spark :
    // Clic droit sur le code ou la declaration de l'objet dans le package explorer, Run As -> Scala Application)
    
    // Chargement du contenu du fichier 'src/main/resources/classification/winequality-red.data'
    // La fonction textFile du SparkContext permet de charger le contenu du fichier
    // Elle retourne un RDD de String que nous travaillerons ensuite
    val rawData = sc.textFile("src/main/resources/classification/winequality-red.data")

    // Objectif suivant : parser les donnees pour en extraire un RDD de LabeledPoint
    // Un LabeledPoint est un vecteur de donnees + un label. Il est utilisé pour les algorithmes supervisés.
    // Le parsing se fait via la fonction parseData dont l'implementation est plus bas.
    val data = rawData.map(parseData)

    // Définition des poids de repartition aleatoire des donnees du dataset pour l'entrainement et les tests
    val weights = Array(.8, .2)

    // Repartir aleatoirement (randomSplit) selon les poids definis
    // Les RDD disposent de la fonction randomSplit pour cela
    val rsplit = data.randomSplit(weights)

    // Affecter le plus gros ensemble de donnees a l'entrainement
    // randomSplit retourne un Array. L'acces aux elements d'un Array se fait ainsi : monTableau(n) avec n>=0
    val trainData = rsplit(0).cache()
    // Affecter le second ensemble de donnees pour les tests
    val testData = rsplit(1)

    // Executer le job Spark (clic droit, Run As -> Scala Application)
    // Apercu de ce que contient le RDD de LabeledPoint
    data.take(3).map( x => printf("Label: %s , Features: %s\n", x.label, x.features) )

    // Repartition des donnees entre jeu d'entrainement et jeu de test
    printf("\nTotal data (%s) = Train data (%s) + Test data (%s)\n", data.count(), trainData.count(), testData.count())

    // Activer la partie A puis B
    partieA(trainData, testData)
    partieB(trainData, testData)
    // Bonus :
//    partieC(trainData, testData)
  }

  def partieA(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) {
    // ------------------------------------------------------------------------

    // Logistic Regression
    // Instancier l'algorithme de regression logistique LogisticRegressionWithLBFGS
    val logreg = new LogisticRegressionWithLBFGS()

    // Et l'entrainer (run) avec le jeu de donnees trainData
    // Note : L'algorithme utilise les parametres par defaut
    val modelLR = logreg.run(trainData)

    // Construire un RDD contenant un Tuple de Double avec dans l'ordre la prediction calculee sur les donnees du dataset de test et le label associe
    // Utiliser le modele pour obtenir la prediction
    // Recuperer le label pour chaque point du jeu de test
    // Construire le Tuple avec ces deux informations. Rappel: un tuple se declare entre parentheses (ex. : ("valeur", 12) )
    val predictionAndLabelLR = testData.map { point =>
      val prediction = modelLR.predict(point.features)
      val label = point.label
      (prediction, label)
    }

    // Afficher les statistiques du modele en appelant evaluateAndPrint
    // Completer la fonction evaluateAndPrint plus bas
    evaluateAndPrint(predictionAndLabelLR, "Logistic Regression")

    // DecisionTree : arbre de decision
    // Les parametres suivants sont definis arbitrairement
    val numClassesDT = 2
    val categoricalFeaturesInfoDT = Map[Int, Int]()
    val impurityDT = "gini"
    val maxDepthDT = 5
    val maxBinsDT = 32

    // Entrainer un modele avec les parametres definis ci dessus pour l'algorithme DecisionTree
    // Pour cela utiliser la fonction trainClassifier de l'objet DecisionTree
    // Un peu comme une methode statique sur une classe Java, utiliser la syntaxe Objet.laFonction(lesParams...)
    val modelDT = DecisionTree.trainClassifier(trainData, numClassesDT, categoricalFeaturesInfoDT, impurityDT, maxDepthDT, maxBinsDT)

    // Construire le RDD comme pour le modele issu de la regression logistique ci dessus
    // La logique est la meme que pour l'algorithme precedent :
    // calcul de la prediction, recuperation du label et construction du Tuple
    val predictionAndLabelDT = testData.map { point =>
      val prediction = modelDT.predict(point.features)
      val label = point.label
      (prediction, label)
    }

    // Evaluer la performance
    evaluateAndPrint(predictionAndLabelDT, "Decision Tree")
  }

  def partieB(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) {

    // ================================================
    // Optimisation d'un modele : recherche des valeurs optimales des parametres
    // Demonstration sur l'algorithme RandomForest (ensemble de DecisionTrees)
    // L'algorithme est sensible au nombre d'arbre ainsi qu'a la profondeur de ceux-ci
    // Nous effectuons un test avec les valeurs de nombre d'arbres : 10, 100, 200, 300
    // Ainsi que les profondeurs : 5, 10, 15, 20
    // Pour chaque combinaison, calculer (et afficher) les metriques avec evaluateAndPrint

    // Parametres "fixes" pour ce test :
    val numClassesRF = 2
    val categoricalFeaturesInfoRF = Map[Int, Int]()
    val featureSubsetStrategyRF = "auto"
    val impurityRF = "gini"
    val maxBinsRF = 32

    // Il faut iterer sur chaque valeur de chaque axe.
    // On peut utiliser pour cela une Sequence (Seq) pour definir les multiples valeurs possibles pour chaque axe
    // Ensuite, il faut parcourir l'ensemble des combinaisons et pour chacune instancier un nouvel algorithme et construire le modele
    // Pour chaque modele, un RDD de Tuple (prediction, label) sera cree pour afficher la performance avec la fonction evaluateAndPrint
    Seq(10, 100, 200, 300).map { tree =>
      Seq(5, 10, 15, 20).map { depth =>
//        ... a) Creation du modele
        val model = RandomForest.trainClassifier(trainData, numClassesRF, categoricalFeaturesInfoRF, tree, featureSubsetStrategyRF, impurityRF, depth, maxBinsRF)
//        ... b) Creation du RDD de Tuples (prediction, label)
        val predictionAndLabelOptim = testData.map { point =>
          val prediction = model.predict(point.features)
          val label = point.label
          (prediction,label)
        }
//        ... c) Evaluation et affichage de la performance
        evaluateAndPrint(predictionAndLabelOptim, "Random Forest with trees=" + tree + " and depth=" + depth)
      }
    }

  }

  /*
   * Implementation d'autres algorithmes de classification.
   * (Bonus)
   */
  def partieC(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) {
    // SVM
    // Entrainer le modele pour l'algorithme SVM (Support Vector Machine) avec le nombre d'iteration defini ci-dessous
    // Utiliser ici aussi l'objet SVMWithSGD et sa fonction train()
    val iters = 100
//    val modelSVM = ...

    // Construire le RDD prediction + label
    val predictionAndLabelSVM = testData.map { point =>
//      val prediction = ...
//      (..., ...)
    }

    // evaluer et afficher
//    evaluateAndPrint(..., "SVM")

    // Random Forest
    val numClassesRF = 2
    val categoricalFeaturesInfoRF = Map[Int, Int]()
    val numTreesRF = 40
    val featureSubsetStrategyRF = "auto"
    val impurityRF = "gini"
    val maxDepthRF = 5
    val maxBinsRF = 32

    // Consignes identiques au DecisionTree appliquees a l'algorithme RandomForest (= ensemble de DecisionTrees)
//    val modelRF = ...

    // Construction du RDD de Tuple
    val predictionAndLabelRF = testData.map { point =>
//      ...
    }

    // Evaluation du modele
//    evaluateAndPrint(..., "Random Forest")

    // Gradient Boosted Trees
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 40
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    // Construction du modele pour l'algorithme GradientBoostedTrees avec les parametres definis ci dessus
    // Utiliser l'objet GradientBoostedTrees et sa fonction train()
//    val modelGBT = ...

    // RDD de Tuple
    val predictionAndLabelGBT = testData.map { point =>
//      ...
    }
    // evaluation
//    evaluateAndPrint(..., "Gradient Boosted Trees")

    // NaiveBayes : Utiliser l'objet NaiveBayes pour entrainer un modele :
//    val modelNB = ...
//    val predictionAndLabelNB = ...
//    evaluateAndPrint(..., "Naive Bayes")

  }

  
  /*
   * Parse les lignes du fichier en LabeledPoint
   * Le label est deduit de la valeur de la derniere colonne de la ligne
   * Les features sont les valeurs precedentes
   * Le LabeledPoint admet un label de type Double et un Vector de Double pour les features
   */
  def parseData(line: String) : LabeledPoint = {
    // Splitter la ligne 'line' sur le caractere ";" afin d'obtenir un Array de String
    val splittedLine = line.split(";")

    // Isoler la derniere valeur du tableau (fonction last)
    // Si cette valeur est strictement inferieure a qualityLevel, le label est 0, sinon 1
    val label = if (splittedLine.last.toInt < qualityLevel) 0 else 1

    // les features sont les elements du tableau moins la derniere valeur (drop right),
    // pour chacun (map sur le tableau resultant) transformés en Double (toDouble)
    val features = splittedLine.dropRight(1).map { _.toDouble }

    // Utilisation de Vectors.dense pour creer un vector de features pour le LabeledPoint
    val vector = Vectors.dense(features)
    new LabeledPoint(label, vector)
  }

  // evalue quelques metriques
  def evaluateAndPrint(pAndl: RDD[(Double, Double)], msg: String) {
    // Compter le nombre d'elements du RDD dont les deux Double sont differents :
    // Filtrer le RDD pAndl pour ne retenir que les elements (qui sont des tuples)
    // pour lesquels la valeur du premier element est differente du deuxieme
    // Compter le nombre d'elements de ce nouvel RDD issu du filtre :
    // nous obtenons le nombre d'erreurs
    val errors = pAndl.filter(r => r._1 != r._2).count.toDouble
    // Diviser ce nombre par le nombre total d'elements dans le RDD pAndl
    val testErr = errors / pAndl.count()
    // Instancie BinaryClassificationMetrics pour obtenir des metriques complementaires avancees (PR et ROC)
    // Voir https://en.wikipedia.org/wiki/Information_retrieval#Performance_and_correctness_measures pour d'eventuels complements theoriques
    val metricsLR = new BinaryClassificationMetrics(pAndl)

    println(msg + "\nTest error=" + testErr + "\t#err= " + errors + "\tROC=" + metricsLR.areaUnderROC() + "\tPR=" + metricsLR.areaUnderPR() + "\n")
  }


  def printCitation() {
    val citation = """
    Wine Quality
    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties
    In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
    
    Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
              [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
              [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib"""
    println(citation)

  }
}