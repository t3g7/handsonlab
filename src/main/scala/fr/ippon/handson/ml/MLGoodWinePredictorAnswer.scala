package fr.ippon.handson.ml

import scala.annotation.varargs
import scala.reflect.runtime.universe
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

/**
 * @author dmartin
 */
object MLGoodWinePredictorAnswer {

  // Seuil de qualite : voir TP precedent
  val qualityLevel = 6

  // Configuration Spark
  val conf = new SparkConf()
    .setAppName("ML with Spark (Hands-on) - Classification with Spark ML")
    .setMaster("local[*]")

  // Contexte Spark
  val sc = new SparkContext(conf)
  // SQL Context
  val sqlContext = new SQLContext(sc)

  def main(args: Array[String]) {

    println("Classification avec Spark ML")
    printCitation()

    // Commencer par executer le job Spark :
    // Clic droit sur le code ou la declaration de l'objet dans le package explorer, Run As -> Scala Application)
    
    val weights = Array(.8, .2)

    // Chargement des donnees brutes CSV du fichier "vin blanc CSV" : 'src/main/resources/classification/winequality-white.csv'
    // Cette fois, le contexte Spark SQL est utilisé pour charger les données du fichier CSV
    val df = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", ";")
      .option("inferSchema", "true")
      .load("src/main/resources/classification/winequality-white.csv")

    // Les données sont décrites dans le fichier 'winequality.names'
    // Il convient d'ajouter une nouvelle colonne décrivant le critere "bon vin" 0/1
    // Nous nommons cette colonne 'good'
    val labelCoder: (Int => Double) = (q: Int) => { if (q < qualityLevel) 0 else 1 }
    val sqlfunc = udf(labelCoder)
    val dataFrame = df.withColumn("good", sqlfunc(col("quality")))

    // Apercu du contenu du dataframe :
    println()
    dataFrame.show(3)

    // Découpage du dataframe : training et test
    // De façon identique au TP précédent, des sous datasets sont crees
    val rSplitDataFrame = dataFrame.randomSplit(weights, 11)

    // Le dataset d'entrainement recoit le plus gros volume de donnees
    val trainDataFrame = rSplitDataFrame(0).cache()

    // Le dataset de test recoit lui le second ensemble de donnees
    val testDataFrame = rSplitDataFrame(1)

    // Activer la partie A puis B selon votre avancement dans le TP
    partieA(trainDataFrame, testDataFrame)
    partieB(trainDataFrame, testDataFrame)

  }

  /*
   * La partie A met en oeuvre un algorithme dans un Pipeline de facon simple
   */
  def partieA(trainDataFrame: DataFrame, testDataFrame: DataFrame) {

    println("\nDataFrame (Spark ML) Random Forest\n")
    // ------ Preparation des elements pour le pipeline...
    // Premier element de type StringIndexer (de type Estimator) prenant en colonne d'entree la colonne 'good' et produisant en sortie la colonne 'label'
    // Instancier un StringIndexer et definir les colonnes d'entree et sortie via les setters correspondant
    val indexer = new StringIndexer()
      .setInputCol("good")
      .setOutputCol("label")

    // Ajout d'un element de type Transformer pour forger le vecteur des features
    // L'implementation VectorAssembler doit etre utilisee.
    // Definir ses donnees en entree (inputCols) : il s'agit d'un Array de toutes les noms de colonnes tels que charges dans le DataFrame
    // Definir la colonne de sortie. Elle sera nommee 'features'
    val vectorAssembler = new VectorAssembler()
      .setInputCols(
        Array(
          "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"))
      .setOutputCol("features")

    // Creation d'un Estimator de type RandomForestClassifier avec les parametres ci-apres :

    val numTreesRF = 40
    val featureSubsetStrategyRF = "auto"
    val impurityRF = "gini"
    val maxDepthRF = 5
    val maxBinsRF = 32

    // Instancier l'algorithme RandomForestClassifier
    // Definir le nom de la colonne portant le label
    // Definir le nom de la colonne portant les features
    val randomForest = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Le Pipeline contient plusieurs objets pouvant recevoir des parametres
    // Ces parametres sont tous regroupes dans une Map
    // En utilisant ParamMap() avec la syntaxe suivante :
    // ParamMap().put(unObjetDuPipeline.unAttribut -> valeurOuVariable, unAutreObjetDuPipeline.unAttribut -> valeurOuVariable, ...)
    // positionner les parametres definis ci-dessus (numTreesRF, ...)
    val paramMap = ParamMap().put(
        randomForest.numTrees -> numTreesRF,
        randomForest.impurity -> impurityRF,
        randomForest.maxDepth -> maxDepthRF,
        randomForest.maxBins -> maxBinsRF,
        randomForest.featureSubsetStrategy -> featureSubsetStrategyRF
        )

    // Creation du Pipeline : definition des etapes (StringIndexer, VectorAssembler, RandomForestClassifier)
    // Instancier un Pipeline et definir les etapes (setStages) sous forme d'un Array contenant
    // les differents composants du pipeline precedemment instancies
    val pipeline = new Pipeline()
      .setStages(
        Array(
          indexer,
          vectorAssembler,
          randomForest))

    // Creation du modele
    // Le modele est produit via la fonction 'fit()' qui admet deux parametres :
    // - le jeu de donnees d'entrainement
    // - la map de parametres
    val modelDF_LR = pipeline.fit(trainDataFrame, paramMap)

    // Calcul des predictions du dataset de test via le modele
    // La fonction 'transform()' est utilisee sur le dataframe de test pour produire les predictions
    val predictions = modelDF_LR.transform(testDataFrame)

    // option : retomber sur les metrics depuis un DataFrame:
    var predsAndlabel = predictions.select("prediction", "label").rdd.map( x => (x.getDouble(0), x.getDouble(1)) )
    var metrics = new BinaryClassificationMetrics(predsAndlabel)
    println("Optional data : Area under ROC: " + metrics.areaUnderROC())

    // Calcul du taux d'erreur (+ erreurs)
    // Calcul du nombre d'erreurs : nombre d'elt du dataframe 'predictions' dont les colonnes 'prediction' et 'label' sont differentes
    // Filtrer le dataframe 'predictions' pour ne retenir que les predictions differentes des labels, compter ce nombre
    val errorsDF_RF = predictions.filter(predictions("prediction") !== predictions("label")).count().toDouble

    // Calcul du ratio nb erreurs / nb total elts du dataframe de test
    val testErrDF_RF = errorsDF_RF / testDataFrame.count()
    
    // Log des performances
    println("DataFrame (Spark ML) Random Forest Test Error = " + testErrDF_RF + ", errors = " + errorsDF_RF)
  }

  /*
   * Cross-Validation
   */
  def partieB(trainDataFrame: DataFrame, testDataFrame: DataFrame) {

    println("\nDataFrame (Spark ML) Random Forest with Cross Validation\n")

    // Reprendre les declarations du StringIndexer, du VectorAssembler, du RandomForestClassifier (pour ce dernier, specifier les parametres ne variant pas) et du Pipeline de la partieA :
    val indexer = new StringIndexer()
      .setInputCol("good")
      .setOutputCol("label")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(
        Array(
          "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"))
      .setOutputCol("features")

    // Instancier l'algorithme RandomForestClassifier
    // Specifier la colonne contenant le label et celle contenant les features
    // Valoriser via les setters les parametres que nous ne ferons pas varier pas dans cette validation : impurity, maxBins et featureSubsetStrategy
    // Utiliser les meme valeurs que celles utilisees en partie A
    val randomForest = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setImpurity("gini")
      .setMaxBins(32)
      .setFeatureSubsetStrategy("auto")

    // Instancier le Pipeline et definir les etapes (setStages())
    val pipeline = new Pipeline()
      .setStages(
        Array(
          indexer,
          vectorAssembler,
          randomForest))

    // Creation via la classe utilitaire ParamGridBuilder du jeu de valeurs pour les parametres cibles (numTrees: 10, 100, 200, 300 et maxDepth: 5, 10, 15, 20)
    // Instancier ParamGridBuilder
    // Ajouter une grille de valeurs pour chaque parametre a faire varier selon la syntaxe :
    // instanceBuilder.addGrid(objetDuPipeline.attribut, Array(val1, val2, ...))
    // Ce, pour chaque attribut devant varier
    // ATTENTION : Temps de calcul longs (~5-15 minutes) avec ces valeurs de parametres
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.numTrees, Array(10, 100, 200, 300))
      .addGrid(randomForest.maxDepth, Array(5, 10, 15, 20))
      .build()

    // Creation d'une instance de l'evaluator BinaryClassificationEvaluator qui servira a identifier la combinaison offrant le meilleur modele
    val evaluator = new BinaryClassificationEvaluator()

    // Instanciation du CrossValidator
    // Definition via les setters de l'Evaluator, de l'Estimator (pipeline) et de la grille de parametres
    val cv = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)

    // Creation du modele ideal via l'instance du CrossValidator
    // La fonction 'fit()' permet ici aussi de calculer le modele.
    // Dans les faits, plusieurs modeles vont etre calcules, et le meilleur sera retourne par la fonction fit()
    val modelDF_CV_RF = cv.fit(trainDataFrame)

    // Calcul des predictions du dataset de test
    // La fonction 'transform()' permet de calculer les predictions pour le dataframe de test
    val cvPredictions = modelDF_CV_RF.transform(testDataFrame)

    // Calcul du taux d'erreur (+ erreurs)
    // Meme regle qu'en partieA mais concernant le DataFrame cvPredictions
    val errors = cvPredictions.filter(cvPredictions("prediction") !== cvPredictions("label")).count().toDouble
    val testErr = errors / testDataFrame.count()

    // Log de la performance du modele
    println("DataFrame (Spark ML) Random Forest with Cross Validation Test Error = " + testErr + ", errors = " + errors)

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