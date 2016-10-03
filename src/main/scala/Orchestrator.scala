
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by saur6410 on 10/2/16.
 */

object Orchestrator {

  def CreateLabeledPointFromInputLine(line: String): LabeledPoint = {
    //ToDo:Change the stub to actual implementation later
    val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
    return pos
  }

  def train(inputFilename: String, args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("SparkGrep").setMaster(args(0))
    val sc = new SparkContext(conf)

    //Get the training data file passed as an argument
    val trainingFileInput = sc.textFile(inputFilename)

    val trainingData = trainingFileInput.map(line => CreateLabeledPointFromInputLine(line))

    //Divide the training data into training and test.
    val positiveSamples = trainingData.filter(point => point.label == 1).randomSplit(Array(0.8, 0.2))
    val negativeSamples = trainingData.filter(point => point.label == 0).randomSplit(Array(0.8, 0.2))

    println ("Positive count:"+(positiveSamples(0).count)+"::"+(positiveSamples(1).count))
    println ("Negative count:"+(negativeSamples(0).count)+"::"+(negativeSamples(1).count))
    val trainingSpamSplit = positiveSamples(0)
    val testSpamSplit = positiveSamples(1)
    val trainingHamSplit = negativeSamples(0)
    val testHamSplit = negativeSamples(1)

    val trainingSplit = trainingSpamSplit ++ trainingHamSplit
    val testSplit = testSpamSplit ++ testHamSplit

    val logisticWithBfgs = Classifier.getAlgorithm("logbfgs", 100, Double.NaN, 0.001)
    val svmWithSGD = Classifier.getAlgorithm("svm", 100, 1, 0.001)
    val logisticWithBfgsPredictsActuals=runClassification(logisticWithBfgs, trainingSplit, testSplit)
    //val svmWithSGDPredictsActuals=runClassification(svmWithSGD, trainingSplit, testSplit)

    //Test the accuracy of the classifier using the test data
    calculateMetrics(logisticWithBfgsPredictsActuals, "Logistic Regression with BFGS")

    //Save the model into a file on HDFS.
  }


  def runClassification(algorithm: GeneralizedLinearAlgorithm[_ <: GeneralizedLinearModel], trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    //Train the classifier using the training data
    //Do n-fold cross-validation to choose the best model.
    val model = algorithm.run(trainingData)
    val predicted = model.predict(testData.map(point => point.features))
    val actuals = testData.map(point => point.label)
    val predictsAndActuals: RDD[(Double, Double)] = predicted.zip(actuals)
    predictsAndActuals
  }

  def calculateMetrics(predictsAndActuals: RDD[(Double, Double)], algorithm: String) {
     val accuracy = 1.0*predictsAndActuals.filter(predActs => predActs._1 == predActs._2).count() / predictsAndActuals.count()
     val binMetrics = new BinaryClassificationMetrics(predictsAndActuals)
     println(s"************** Printing metrics for $algorithm ***************")
     println(s"Area under ROC ${binMetrics.areaUnderROC}")
     //println(s"Accuracy $accuracy")
     val metrics = new MulticlassMetrics(predictsAndActuals)
     val f1=metrics.fMeasure
     println(s"F1 $f1")
     println(s"Precision : ${metrics.precision}")
     println(s"Confusion Matrix \n${metrics.confusionMatrix}")
     println(s"************** ending metrics for $algorithm *****************")
     }

  def getModel():Unit = {
    //Load the classifier from the file.

    //return the model
    }

  def predict():Unit = {
    //Take the model and the input vector as the arguments

    //return the prediction
  }


}