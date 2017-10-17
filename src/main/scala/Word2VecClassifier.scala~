
//package isr.project
package org.apache.spark.mllib.linalg

import java.io.IOException

import isr.project.{IdfFeatureGenerator, Tweet}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, MultilabelMetrics}
import org.apache.spark.mllib.feature.{HashingTF, IDFModel, Word2Vec, Word2VecModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Try

//case class Tweet(id: String, tweetText: String, label: Option[Double] = None)

object Word2VecClassifier{




  val _threshold = 0.25
  var _lrModelFilename = "data/lrclassifier.model"
  var _numberOfClasses = 9
  var _word2VecModelFilename = "data/word2vec.model"





  def train(tweets: RDD[Tweet], sc:SparkContext): (Word2VecModel,LogisticRegressionModel) = {
    val bcNumberOfClasses = sc.broadcast(_numberOfClasses)
    val bcWord2VecModelFilename = sc.broadcast(_word2VecModelFilename)
    val bcLRClassifierModelFilename = sc.broadcast(_lrModelFilename)

    def cleanHtml(str: String) = str.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", "")

    def cleanTweetHtml(sample: Tweet) = sample copy (tweetText = cleanHtml(sample.tweetText))

    val cleanTrainingTweets = tweets map cleanTweetHtml

    // Words only
    def cleanWord(str: String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W", "")).reduceOption((x, y) => s"$x $y")

    def wordOnlySample(sample: Tweet) = sample copy (tweetText = cleanWord(sample.tweetText).getOrElse(""))

    val wordOnlyTrainSample = cleanTrainingTweets map wordOnlySample

    // Word2Vec
    val samplePairs = wordOnlyTrainSample.map(s => s.id -> s)
    val reviewWordsPairs: RDD[(String, Iterable[String])] = samplePairs.mapValues(_.tweetText.split(" ").toIterable)

    val word2vecModel = new Word2Vec().fit(reviewWordsPairs.values)
    // commented out to run on our system
    //word2vecModel.save(sc, bcWord2VecModelFilename.value)


    def wordFeatures(words: Iterable[String]): Iterable[Vector] = words.map(w => Try(word2vecModel.transform(w))).filter(_.isSuccess).map(x => x.get)

    def avgWordFeatures(wordFeatures: Iterable[Vector]): Vector = Vectors.fromBreeze(wordFeatures.map(_.toBreeze).reduceLeft((x, y) => x + y) / wordFeatures.size.toDouble)

    def filterNullFeatures(wordFeatures: Iterable[Vector]): Iterable[Vector] = if (wordFeatures.isEmpty) wordFeatures.drop(1) else wordFeatures

    // Create feature vectors
    val wordFeaturePairTrain = reviewWordsPairs mapValues wordFeatures
    //val intermediateVectors = wordFeaturePair.mapValues(x => x.map(_.asBreeze))
    val inter2Train = wordFeaturePairTrain.filter(!_._2.isEmpty)
    val avgWordFeaturesPairTrain = inter2Train mapValues avgWordFeatures
    val featuresPairTrain = avgWordFeaturesPairTrain join samplePairs mapValues {
      case (features, Tweet(id, tweetText, label)) => LabeledPoint(label.get, features)
    }
    val trainingSet = featuresPairTrain.values

    // Classification
    println("String Learning and evaluating models")

    val logisticRegressionModel = GenerateOptimizedModel(trainingSet,bcNumberOfClasses.value)
    // commented out to run on our machines
    //logisticRegressionModel.save(sc, bcLRClassifierModelFilename.value)
    return (word2vecModel,logisticRegressionModel)
  }

  def trainIdfClassifer(tweets: RDD[Tweet], sc: SparkContext) = {
    val bcNumberOfClasses = sc.broadcast(_numberOfClasses)
    val bcWord2VecModelFilename = sc.broadcast(_word2VecModelFilename)
    val bcLRClassifierModelFilename = sc.broadcast(_lrModelFilename)

    def cleanHtml(str: String) = str.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", "")

    def cleanTweetHtml(sample: Tweet) = sample copy (tweetText = cleanHtml(sample.tweetText))

    val cleanTrainingTweets = tweets map cleanTweetHtml

    // Words only
    def cleanWord(str: String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W", "")).reduceOption((x, y) => s"$x $y")

    def wordOnlySample(sample: Tweet) = sample copy (tweetText = cleanWord(sample.tweetText).getOrElse(""))

    val wordOnlyTrainSample = cleanTrainingTweets map wordOnlySample

    // Word2Vec
    val samplePairs = wordOnlyTrainSample.map(s => s.id -> s)

    val idfGenerator = new IdfFeatureGenerator()
    val (idfModel, hashingModel) = idfGenerator.InitializeIDF(samplePairs.map(x => x._2))

    val wordIdfFeaturesTrain = samplePairs.mapValues(t => t.tweetText).mapValues(tweets => GetIdfForWord(tweets, idfModel, hashingModel))

    val featuresPairTrain = wordIdfFeaturesTrain join samplePairs mapValues {
      case (features, Tweet(id, tweetText, label)) => LabeledPoint(label.get, features._2)
    }
    val trainingSet = featuresPairTrain.values

    // Classification
    println("String Learning and evaluating models for IDF features.")

    val logisticRegressionModel = GenerateOptimizedModel(trainingSet, bcNumberOfClasses.value)
    // commented out to run on our machines
    //logisticRegressionModel.save(sc, bcLRClassifierModelFilename.value)
    (idfModel, hashingModel, logisticRegressionModel)
  }

  def GetIdfForWord(tweetText: String, idfModel: IDFModel, hashingModel: HashingTF): (String, Vector) = {
    if (idfModel == null)
      throw new Exception("IDF dictionary not initialized")
    //Everything is fine. Return the IDF value of the word.
    val features = hashingModel.transform(tweetText.split(" "))
    val wordFeatures = idfModel.transform(features)
    return (tweetText, wordFeatures)
  }

  def predict(tweets: RDD[Tweet], sc: SparkContext, w2vModel: Word2VecModel, lrModel: LogisticRegressionModel): (RDD[Tweet], RDD[(Double, Double)]) = {
    //val sc = new SparkContext()

    //Broadcast the variables
    //val bcNumberOfClasses = sc.broadcast(_numberOfClasses)
    //val bcWord2VecModelFilename = sc.broadcast(_word2VecModelFilename)
    //val bcLRClassifierModelFilename = sc.broadcast(_lrModelFilename)

    def cleanHtml(str: String) = str.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", "")

    def cleanTweetHtml(sample: Tweet) = sample copy (tweetText = cleanHtml(sample.tweetText))

    val cleanTestTweets = tweets map cleanTweetHtml
    val word2vecModel = w2vModel //Word2VecModel.load(sc, bcWord2VecModelFilename.value)
    //println(s"Model file found:${bcWord2VecModelFilename.value}. Loading model.")
    //println("Finished Training")
    //println(word2vecModel.transform("hurricane"))

    // Words only
    def cleanWord(str: String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W", "")).reduceOption((x, y) => s"$x $y")

    def wordOnlySample(sample: Tweet) = sample copy (tweetText = cleanWord(sample.tweetText).getOrElse(""))

    def wordFeatures(words: Iterable[String]): Iterable[Vector] = words.map(w => Try(word2vecModel.transform(w))).filter(_.isSuccess).map(x => x.get)

    def avgWordFeatures(wordFeatures: Iterable[Vector]): Vector = Vectors.fromBreeze(wordFeatures.map(_.toBreeze).reduceLeft((x, y) => x + y) / wordFeatures.size.toDouble)

    def filterNullFeatures(wordFeatures: Iterable[Vector]): Iterable[Vector] = if (wordFeatures.isEmpty) wordFeatures.drop(1) else wordFeatures

    val wordOnlyTestSample = cleanTestTweets map wordOnlySample
    val samplePairsTest = wordOnlyTestSample.map(s => s.id -> s)
    val reviewWordsPairsTest: RDD[(String, Iterable[String])] = samplePairsTest.mapValues(_.tweetText.split(" ").toIterable)
    val wordFeaturePairTest = reviewWordsPairsTest mapValues wordFeatures
    val inter2Test = wordFeaturePairTest.filter(!_._2.isEmpty)
    val avgWordFeaturesPairTest = inter2Test mapValues avgWordFeatures
    val featuresPairTest = avgWordFeaturesPairTest join samplePairsTest mapValues {
      case (features, Tweet(id, tweetText, label)) => (Tweet(id, tweetText, label), features)
    }
    val testSet = featuresPairTest.values



    val logisticRegressionModel = lrModel //LogisticRegressionModel.load(sc, bcLRClassifierModelFilename.value)
    //println(s"Classifier Model file found:$bcLRClassifierModelFilename. Loading model.")

    val start = System.currentTimeMillis()
    val logisticRegressionPredictions = testSet.map { case (Tweet(id, tweetText, label), features) =>
      val prediction = logisticRegressionModel.predict(features)
      Tweet(id, tweetText, Option(prediction))
    }
    val logisticRegressionPredLabel = testSet.map { case (Tweet(id, tweetText, label), features) =>
      val prediction = logisticRegressionModel.predict(features)
      (prediction, label.getOrElse(9999999999.0))
    }
    println("<---- done")
    val end = System.currentTimeMillis()
    println(s"Took ${(end - start) / 1000.0} seconds for Prediction.")

    return (logisticRegressionPredictions, logisticRegressionPredLabel)
  }

  def predictForIDFClassifer(tweets: RDD[Tweet], sc: SparkContext, idfModel: IDFModel, hashingModel: HashingTF, lrModel: LogisticRegressionModel): (RDD[Tweet], RDD[(Double, Double)]) = {
    //val sc = new SparkContext()

    //Broadcast the variables
    //val bcNumberOfClasses = sc.broadcast(_numberOfClasses)
    //val bcWord2VecModelFilename = sc.broadcast(_word2VecModelFilename)
    //val bcLRClassifierModelFilename = sc.broadcast(_lrModelFilename)

    def cleanHtml(str: String) = str.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", "")

    def cleanTweetHtml(sample: Tweet) = sample copy (tweetText = cleanHtml(sample.tweetText))

    val cleanTestTweets = tweets map cleanTweetHtml

    // Words only
    def cleanWord(str: String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W", "")).reduceOption((x, y) => s"$x $y")

    def wordOnlySample(sample: Tweet) = sample copy (tweetText = cleanWord(sample.tweetText).getOrElse(""))

    val wordOnlyTestSample = cleanTestTweets map wordOnlySample
    val samplePairsTest = wordOnlyTestSample.map(s => s.id -> s)


    val wordIdfFeaturesTest = samplePairsTest.mapValues(t => t.tweetText).mapValues(tweets => GetIdfForWord(tweets, idfModel, hashingModel))
    val featuresPairTest = wordIdfFeaturesTest join samplePairsTest mapValues {
      case (features, Tweet(id, tweetText, label)) => (Tweet(id, tweetText, label), features._2)
    }
    val testSet = featuresPairTest.values
 //   testSet.cache()



    val logisticRegressionModel = lrModel

    val start = System.currentTimeMillis()
    val logisticRegressionPredictions = testSet.map { case (Tweet(id, tweetText, label), features) =>
      val prediction = logisticRegressionModel.predict(features)
      Tweet(id, tweetText, Option(prediction))
    }
    val logisticRegressionPredLabel = testSet.map { case (Tweet(id, tweetText, label), features) =>
      val prediction = logisticRegressionModel.predict(features)
      (prediction, label.getOrElse(9999999999.0))
    }
    println("<---- done")
    val end = System.currentTimeMillis()
    println(s"Took ${(end - start) / 1000.0} seconds for Prediction.")

    return (logisticRegressionPredictions, logisticRegressionPredLabel)
  }

  def run(args: Array[String], delimiter: Char) {

    if (args.length < 4) {
      System.err.println("Usage: SparkGrep <host> <training_file> <test_file> <numberofClasses>")
      System.exit(1)
    }

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val trainFilename = args(1)
    val testFilename = args(2)
    _numberOfClasses = args(3).toInt
    val partitionCount = 128
    val trainingPartitionCount = 8



    def printRDD(xs: RDD[_]) {
      println("--------------------------")
      xs take 5 foreach println
      println("--------------------------")
    }

    val conf = new SparkConf(false).setMaster(args(0)).setAppName("Word2Vec")
    val sc = new SparkContext(conf)

    //Broadcast the variables
    val bcNumberOfClasses = sc.broadcast(_numberOfClasses)
    val bcWord2VecModelFilename = sc.broadcast(_word2VecModelFilename)
    val bcLRClassifierModelFilename = sc.broadcast(_lrModelFilename)

    // Load
    val trainPath =  trainFilename
    val testPath = testFilename

    // Load text
    def skipHeaders(idx: Int, iter: Iterator[String]) = if (idx == 0) iter else iter

    val trainFile = sc.textFile(trainPath, trainingPartitionCount) mapPartitionsWithIndex skipHeaders map (l => l.split(delimiter))
    val testFile = sc.textFile(testPath, partitionCount) mapPartitionsWithIndex skipHeaders map (l => l.split(delimiter))

    //trainFile.cache()


    // To sample
    def toTweet(segments: Array[String]) = segments match {
      case Array(label, tweetText) => Tweet(java.util.UUID.randomUUID.toString, tweetText, Some(label.toDouble))
    }

    val trainingTweets = trainFile map toTweet
    val testTweets = testFile map toTweet

    // Clean Html
    def cleanHtml(str: String) = str.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", "")

    def cleanTweetHtml(sample: Tweet) = sample copy (tweetText = cleanHtml(sample.tweetText))

    val cleanTrainingTweets = trainingTweets map cleanTweetHtml
    val cleanTestTweets = testTweets map cleanTweetHtml

    // Words only
    def cleanWord(str: String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W", "")).reduceOption((x, y) => s"$x $y")

    def wordOnlySample(sample: Tweet) = sample copy (tweetText = cleanWord(sample.tweetText).getOrElse(""))

    val wordOnlyTrainSample = cleanTrainingTweets map wordOnlySample
    val wordOnlyTestSample = cleanTestTweets map wordOnlySample

    // Word2Vec
    val samplePairs = wordOnlyTrainSample.map(s => s.id -> s).cache()
    val reviewWordsPairs: RDD[(String, Iterable[String])] = samplePairs.mapValues(_.tweetText.split(" ").toIterable)
    println("Start Training Word2Vec --->")

    var word2vecModel:Word2VecModel = null

    //reviewWordsPairs.repartition(partitionCount)
    reviewWordsPairs.cache()

    try {
      word2vecModel = new Word2Vec().fit(reviewWordsPairs.values)
      //word2vecModel = Word2VecModel.load(sc, bcWord2VecModelFilename.value)
      println(s"Model file found:${bcWord2VecModelFilename.value}. Loading model.")
    }
    catch{
      case ioe: IOException =>
          println(s"Model not found at ${bcWord2VecModelFilename.value}. Creating model.")
          word2vecModel = new Word2Vec().fit(reviewWordsPairs.values)
        //word2vecModel.save(sc, bcWord2VecModelFilename.value);
          println(s"Saved model as ${bcWord2VecModelFilename.value} .")
    }


    println("Finished Training")
    println(word2vecModel.transform("hurricane"))
    //println(word2vecModel.findSynonyms("shooting", 4))

    def wordFeatures(words: Iterable[String]): Iterable[Vector] = words.map(w => Try(word2vecModel.transform(w))).filter(_.isSuccess).map(x => x.get)

    def avgWordFeatures(wordFeatures: Iterable[Vector]): Vector = Vectors.fromBreeze(wordFeatures.map(_.toBreeze).reduceLeft((x, y) => x + y) / wordFeatures.size.toDouble)

    def filterNullFeatures(wordFeatures: Iterable[Vector]): Iterable[Vector] = if (wordFeatures.isEmpty) wordFeatures.drop(1) else wordFeatures

    // Create feature vectors
    val wordFeaturePairTrain = reviewWordsPairs mapValues wordFeatures
    //val intermediateVectors = wordFeaturePair.mapValues(x => x.map(_.asBreeze))
    val inter2Train = wordFeaturePairTrain.filter(!_._2.isEmpty)
    val avgWordFeaturesPairTrain = inter2Train mapValues avgWordFeatures
    val featuresPairTrain = avgWordFeaturesPairTrain join samplePairs mapValues {
      case (features, Tweet(id, tweetText, label)) => LabeledPoint(label.get, features)
    }
    val trainingSet = featuresPairTrain.values

    // Classification
    println("String Learning and evaluating models")
    //val Array(x_train, x_test) = trainingSet.randomSplit(Array(0.7, 0.3))
    // Run training algorithm to build the model


    trainingSet.repartition(trainingPartitionCount)



    val samplePairsTest = wordOnlyTestSample.map(s => s.id -> s).cache()
    val reviewWordsPairsTest : RDD[(String, Iterable[String])] = samplePairsTest.mapValues(_.tweetText.split(" ").toIterable)
    val wordFeaturePairTest = reviewWordsPairsTest mapValues wordFeatures
    val inter2Test = wordFeaturePairTest.filter(!_._2.isEmpty)
    val avgWordFeaturesPairTest = inter2Test mapValues avgWordFeatures
    val featuresPairTest = avgWordFeaturesPairTest join samplePairsTest mapValues {
      case (features, Tweet(id, tweetText, label)) => (LabeledPoint(label.get, features), tweetText)
    }
    val testSet = featuresPairTest.values
    testSet.cache()
    //testSet.repartition(partitionCount)

    //val trainingRDD = trainingSet.toJavaRDD()
    //val svmModel = SVMMultiClassOVAWithSGD.train(trainingRDD, 100 )
    // Compute raw scores on the test set.

    //import spark.implicits._

    //val (logisticRegressionPredictions, start) = NFoldBasedWord2VecClassifier.GeneratePredictions(trainingSet, testSet, sc)
    val (logisticRegressionPredictions, start) = GeneratePredictions(trainingSet, testSet, sc, bcNumberOfClasses.value, bcLRClassifierModelFilename.value)
    val classZeroPredictionCount = logisticRegressionPredictions.filter(pred => pred._1 == 0.0).count()
    println(s"Zero class count =  ${classZeroPredictionCount}")
    GenerateClassifierMetrics(logisticRegressionPredictions, "Logistic Regression", bcNumberOfClasses.value)

    println("<---- done")
    val end = System.currentTimeMillis()
    println(s"Took ${(end-start)/1000.0} seconds for Prediction.")
    //Thread.sleep(10000)
  }

  def GeneratePredictions(trainingData: RDD[LabeledPoint],
                          testData: RDD[(LabeledPoint, String)],
                          sc:SparkContext,
                          bcNumberOfClasses:Int,
                          bcLRClassifierModelFilename:String):
  (RDD[(Double, Double)], Long) =
    {
      var logisticRegressionModel: LogisticRegressionModel = null

      try {
      logisticRegressionModel =  LogisticRegressionModel.load(sc, bcLRClassifierModelFilename)
      println(s"Classifier Model file found:${bcLRClassifierModelFilename}. Loading model.")
    }
    catch{
      case ioe: IOException =>
          println(s"Classifier Model not found at ${bcLRClassifierModelFilename}. Creating model.")
          logisticRegressionModel =  GenerateOptimizedModel(trainingData, bcNumberOfClasses)
        //logisticRegressionModel.save(sc, bcLRClassifierModelFilename);
          println(s"Saved classifier  model as ${bcLRClassifierModelFilename} .")
    }


    val start = System.currentTimeMillis()
      /*val logisticRegressionPredictions = testData.map { case LabeledPoint(label, features) =>
        val prediction = logisticRegressionModel.predict(features)
        (prediction, label)
      }*/

      val logisticRegressionPredictions = testData
        .map { case (LabeledPoint(label, features), tweetText) =>
          val (prediction, probabilities) = ClassificationUtility
            .predictPoint(features, logisticRegressionModel)
          (prediction, label, probabilities, tweetText)
        }




      val highProbabilityMisclassifications = logisticRegressionPredictions.filter(pred => pred._3.max > _threshold && pred._1 != pred._2)
      val lowProbabilityClassifications = logisticRegressionPredictions.filter(pred => pred._3.max < _threshold && pred._1 == pred._2)
      //val eric = logisticRegressionPredictions.filter(p => p._3.max > 0.5)
      (logisticRegressionPredictions.map(pred => (if (pred._3.max == pred._3(pred._1.toInt) && pred._3(pred._1.toInt) > _threshold) pred._1 else 0.0, pred._2)), start)
    }

  def GenerateOptimizedModel(trainingData: RDD[LabeledPoint], bcNumberOfClasses: Int)
  : LogisticRegressionModel = {

    /*val foldCount = 10
    //Break the trainingData into n-folds
    for (i <- 1 to foldCount) {
      val setSize = trainingData.count()
      val subTrainData = trainingData.fo

    }*/
    val lrClassifier = new LogisticRegressionWithLBFGS()
    lrClassifier.optimizer.setConvergenceTol(0.01)
    //lrClassifier.optimizer.setNumIterations(75)

    lrClassifier
      .setNumClasses(bcNumberOfClasses)
      .run(trainingData)
  }

  def GenerateClassifierMetrics(predictionAndLabels: RDD[(Double, Double)]
                                ,classifierType : String,
                                bcNumberOfClasses:Int)
  : Unit = {
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val otherMetrics = new MultilabelMetrics(predictionAndLabels.map(elem => (Array(elem._1), Array(elem._2))))
    //val uniqueLabels = predictionAndLabels.map(x => x._1).

    for (i <- metrics.labels) {
    //for (i <- uniqueLabels) {
      val classLabel = i
      println(s"\n***********   Class:$classLabel   *************")
      println(s"F1 Score:${metrics.fMeasure(classLabel)}")
      println(s"True Positive:${metrics.truePositiveRate(classLabel)}")
      println(s"False Positive:${metrics.falsePositiveRate(classLabel)}")
      println(s"Precision:${metrics.precision(classLabel)}")
      println(s"Recall:${metrics.recall(classLabel)}")
    }
    println(s"\nConfusion Matrix \n${metrics.confusionMatrix}")
    val f1Measure = metrics.weightedFMeasure
    val precision = metrics.weightedPrecision
    val recall = metrics.weightedRecall
    val macrof1 = metrics.labels.map(lab => metrics.fMeasure(lab)).sum/metrics.labels.length
    // this (otherMetrics.microF1Measure) gives the exact same as metrics.fmeasure when I was testing it
    val microf1 = otherMetrics.microF1Measure
    println(s"\n***********   Classifier Results for $classifierType   *************")
    println(s"Weighted F1-Measure = $f1Measure")
    println(s"Weighted Precision = $precision")
    println(s"Weighted Recall = $recall")
    println(s"Macro F1 = $macrof1")
    println(s"Micro F1 = $microf1")

    println(s"\n***********   End of Classifier Results for $classifierType   *************")
  }
 }



