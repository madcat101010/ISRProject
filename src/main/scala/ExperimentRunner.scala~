import isr.project.{SparkGrep, Tweet}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Word2VecClassifier
import org.apache.spark.rdd.RDD

/**
  * Created by Eric on 12/23/2016.
  */
object ExperimentRunner {

  val training_partitions = 64
  val testing_partitions = 64

  def main(args: Array[String]): Unit = {
    // for the product datasets
    val base_dirs = Seq(
      "data/product_data/uol-electronic"
      //"data/product_data/uol-non-electronic"
      //"data/product_data/uol-book"
      //"data/mega_set/"
    )
    val suffix = Seq(1).toList.map(_.toString)


    // for the small tweet datasets
    /*val base_dirs = Seq("data/accuracy_experiment/data")
    val suffix = (2 to 10).toList.map(_.toString + ".rw")*/

    // if running local we must set a have this uncommented, on the cluster we want the other sc
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("ExperimentOrchestration")
    val sc = new SparkContext(conf)
    //val sc = new SparkContext()
    //turnOffLogging(sc)
    var allMetrics = List[List[List[ExperimentalMetrics]]]()
    var allIDFMetrics = List[List[List[ExperimentalMetrics]]]()
    for (dir <- base_dirs) {
      // for the product datasets
      val experimentalMetrics = run_w2v_experiments(dir, "/train", "/test", suffix, 1, sc)
      //val idfExperimentMetrics = run_idf_experiments(dir, "/train", "/test", suffix, 3, sc)
      // for the tweet datasets
      //val experimentalMetrics = run_w2v_experiments(dir, "/train_data", "/test_data", suffix, 3, sc)
      //val idfExperimentMetrics = run_idf_experiments(dir,"/train_data","/test_data",suffix,3,sc)
      allMetrics = allMetrics :+ experimentalMetrics
      //allIDFMetrics = allIDFMetrics :+ idfExperimentMetrics
    }
    println("NOW W2V METRICS")
    printMetricResults(base_dirs, allMetrics)
    //println("NOW IDF METRICS")
    //printMetricResults(base_dirs, allIDFMetrics)

  }

  private def printMetricResults(base_dirs: Seq[String], allMetrics: List[List[List[ExperimentalMetrics]]]) = {
    try {
      for ((dir, idx) <- base_dirs.zipWithIndex) {
        println(s"####Experiment dir $dir####")
        val experimentMetrics = allMetrics(idx)
        println(s"###Per set ###")
        for (setMetric <- experimentMetrics) {
          println(ExperimentalMetrics.header())
          for (met <- setMetric) {
            println(met.toString())
          }
        }
        println(s"## ALL ##")
        println(ExperimentalMetrics.header())
        for (metric <- experimentMetrics.flatten) {
          println(metric.toString())
        }

      }
    }
    catch {
      case e: Exception =>
        println("exception")
        e.printStackTrace()
    }
  }

  private def turnOffLogging(sc: SparkContext) = {
    sc.setLogLevel(Level.ERROR.toString)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getRootLogger.setLevel(Level.ERROR)
  }

  def run_w2v_experiments(dir: String, trainFile: String, testFile: String, suffixes: List[String], timesToRunEach: Int, sc: SparkContext): List[List[ExperimentalMetrics]] = {
    var experimentSet = List[List[ExperimentalMetrics]]()
    for (suffix <- suffixes) {
      val trainFName = dir + trainFile + suffix
      val testFName = dir + testFile + suffix
      var oneSet = List[ExperimentalMetrics]()
      for (i <- 1 to timesToRunEach) {
        println(s"Running ${trainFName} ${testFName}")
        val m = scala.collection.mutable.Map[String, Double]()
        val trainTweetsRDD = SparkGrep.getTweetsFromFile(trainFName, m, sc).cache().repartition(64)

        val testTweetsRDD = SparkGrep.getTweetsFromFile(testFName, m, sc).cache().repartition(64)
        //trainTweetsRDD.repartition(64)
        //testTweetsRDD.repartition(64)
        //trainTweetsRDD.cache()
        //testTweetsRDD.cache()
        SparkGrep.SetupWord2VecField(trainFName, trainTweetsRDD)
        val (word2VecModel, logisticRegressionModel, trainTime) = SparkGrep.PerformTraining(sc, trainTweetsRDD)
        val predictstart = System.currentTimeMillis()
        val (predictionTweets, predictionLabel) = performPrediction(sc, word2VecModel, logisticRegressionModel, testTweetsRDD)
        val predicted_tweets = predictionTweets.collect()
        val predict_stop = System.currentTimeMillis()
        val predictTime = (predict_stop - predictstart) / 1000.0
        val Metrics = MetricsCalculator.GenerateClassifierMetrics(predictionLabel)
        Metrics.trainTime = trainTime
        Metrics.predictTime = predictTime
        oneSet = oneSet :+ Metrics
      }
      experimentSet = experimentSet :+ oneSet

    }
    return experimentSet
  }

  def performPrediction(sc: SparkContext, word2VecModel: Word2VecModel, logisticRegressionModel: LogisticRegressionModel, testTweetRdd: RDD[Tweet]): (RDD[Tweet], RDD[(Double, Double)]) = {
    val (predictionTweets, predictionLabel) = Word2VecClassifier.predict(testTweetRdd, sc, word2VecModel, logisticRegressionModel)
    (predictionTweets, predictionLabel)
  }

  def run_idf_experiments(dir: String, trainFile: String, testFile: String, suffixes: List[String], timesToRunEach: Int, sc: SparkContext): List[List[ExperimentalMetrics]] = {
    var experimentSet = List[List[ExperimentalMetrics]]()
    for (suffix <- suffixes) {
      val trainFName = dir + trainFile + suffix
      val testFName = dir + testFile + suffix
      var oneSet = List[ExperimentalMetrics]()
      for (i <- 1 to timesToRunEach) {
        println(s"Running ${trainFName} ${testFName}")
        val m = scala.collection.mutable.Map[String, Double]()
        val trainTweets = SparkGrep.getTweetsFromFile(trainFName, m, sc)
        val trainTweetsRDD = trainTweets
        val testTweets = SparkGrep.getTweetsFromFile(testFName, m, sc)
        val testTweetsRDD = testTweets
        SparkGrep.SetupWord2VecField(trainFName, trainTweets)
        //This is how the IDF based classifier would run.
        val (idfModel, hashingTfModel, idfLrModel, idfTrainTime) = SparkGrep.PerformIDFTraining(sc, trainTweetsRDD)
        val (predictionIDFTweets, predictionIDFLabels) = Word2VecClassifier.predictForIDFClassifer(testTweetsRDD, sc, idfModel, hashingTfModel, idfLrModel)

        val Metrics = MetricsCalculator.GenerateClassifierMetrics(predictionIDFLabels)
        Metrics.trainTime = idfTrainTime
        oneSet = oneSet :+ Metrics
      }
      experimentSet = experimentSet :+ oneSet

    }
    return experimentSet
  }

}
