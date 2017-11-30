/*


*/

package isr.project

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Word2VecClassifier
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.hbase.client.HBaseAdmin
import scala.util.Random
import scala.collection.mutable.ArrayBuffer


object SparkGrep {
  def tweetchange(tweet: Tweet): Tweet = {
    if (tweet.label.get == 0.0) {
      return Tweet(tweet.id, tweet.tweetText, Option(9.0))
    }
    return Tweet(tweet.id, tweet.tweetText, tweet.label)
  }

//SparkGrep <train/classify/label> <webpage/tweet> <srcTableName> <destTableName> <event name> <collection name> <class1Name> <class2Name> [class3Name] [...]
  def main(args: Array[String]) {
		if(args.length >= 7){
			//ensure correct usage
			if(args(0) != "train" && args(0) != "classify" && args(0) != "label"){
				System.err.println("Usage Error: args(0) != 'train' or 'classify' or 'label'");
      	System.exit(1);
			}
			if(args(1) != "webpage" && args(1) != "tweet"){
				System.err.println("Usage Error: args(1) != 'tweet' or 'webpage'");
      	System.exit(1);
			}
			
			//load collection name and class mapping for that collection
			//class 0.0 is "Can't Classify!" always, so start with class 1.0
			val eventName = args(4)
			val collectionName = args(5);
			var classCount = 1;
			for( x <- 6 to (args.length-1) ){
				DataWriter.mapLabel( (x-5).toDouble, args(x) );
				classCount = classCount + 1;
			}
			Word2VecClassifier._numberOfClasses = (classCount).toInt; 
			println("Number of classes is: " + Word2VecClassifier._numberOfClasses.toInt);
			var tableNameSrc = args(2);
			//var tableNameSrc = tableName;
			var tableNameDest = args(3);
			//train or classify
			val start = System.currentTimeMillis()
			if(args(0) == "train"){
				val conf = new SparkConf()
				    .setMaster("local[*]")
				    .setAppName("CLA-ModelTraining")
				val sc = new SparkContext(conf);
				if(args(1) == "tweet"){
					var tweetTrainingFile = ("./data/training/" + eventName + "_tweet_training.csv");	//TODO:ensure file ending is good
					var tweetTestingFile = ("./data/testing/" + eventName + "_tweet_testing.csv");	//TODO:ensure file ending is good
					Word2VecClassifier._lrModelFilename = "./data/" + eventName + "_tweet_lr.model";
					Word2VecClassifier._word2VecModelFilename = "./data/" + eventName + "_tweet_w2v.model";
					TrainTweetModels(tweetTrainingFile, tweetTestingFile, sc);	//TODO: File formatting must match when we make the training/testing files...
				}
				else if(args(1) == "website"){
					var websiteTrainingFile = ("./data/training/" + eventName + "_webpage_training.csv");	//TODO:ensure file ending is good
					var websiteTestingFile = ("./data/testing/" + eventName + "_webpage_testing.csv");
					TrainWebsiteModelsBasedTweet(tableNameSrc,websiteTrainingFile, websiteTestingFile, sc)  //TODO: Don't combine csv file anymore... don't random pick train:test data
				}
			}
			else if(args(0) == "classify"){
				Logger.getLogger("org").setLevel(Level.OFF)
				Logger.getLogger("akka").setLevel(Level.OFF)
				val conf = new SparkConf()
		        .setMaster("local[*]")
		        .setAppName("CLA-Classifying")
				val sc = new SparkContext(conf)

				if(args(1) == "tweet"){
		    	DataRetriever.retrieveTweets(eventName, collectionName, 1000, tableNameSrc, tableNameDest, sc)
				}
				else if(args(1) == "website"){
		    	DataRetriever.retrieveWebpages(eventName, collectionName, 50, tableNameSrc, tableNameDest, sc)
				}
			}
			else if(args(0) == "label"){
				val conf = new SparkConf()
				.setMaster("local[*]")
				.setAppName("HBaseProductExperiments")
				val sc = new SparkContext(conf)
				if(args(1) == "website"){
					println("TODO: Labeling Website Training Data")
				}
				else{
					println("Labeling Tweet Training Data")
					val trainingTweets = DataRetriever.getTrainingTweets(sc, args(2), args(5))
					trainingTweets.map(tweet => tweetToCSVLine(tweet)).saveAsTextFile("./data/training/" + args(4) + "_tweet_training.csv")
				}
			}
  		val end = System.currentTimeMillis()
  		println(s"Took ${(end - start) / 1000.0} seconds for the whole process.")
			System.exit(0)
		}
    else{
      System.err.println("Usage: SparkGrep <train/classify/label> <webpage/tweet> <srcTableName> <destTableName> <event name> <collection name> <class1Name> <class2Name> [class3Name] [...]")
			System.err.println("Usage Note: files should be in the ./data/ directory i.e. './data/collection1_trainingfile'")
      System.exit(1)
    }

  }

	def tweetToCSVLine(tweet:Tweet):String={
		return tweet.id + "," + tweet.tweetText + "," + tweet.label.get.toString
	}
  
  def TrainTweetModels(trainFile: String, testFile: String, sc: SparkContext): Unit = {
    println("Training Tweet models")

		//load training files
    var labelMap = scala.collection.mutable.Map[String,Double]()
    val training_partitions = 8
    val testing_partitions = 8

		//val trainTweets = getTweetsFromFile(trainFile, labelMap, sc).collect()
    //val testTweets = getTweetsFromFile(testFile, labelMap, sc).collect()
		//eclipsedatasample1

		//randomly pick out tweets for testing and training
		val allTweets = getTweetsFromFile(trainFile, labelMap, sc).collect().toBuffer
		println("*********")
		var trainTweetsB = ArrayBuffer[Tweet]()
		var testTweetsB = ArrayBuffer[Tweet]()
		for((k,v) <- labelMap){
			val singleClassTweets = allTweets.filter(y => y.label == labelMap.get(k))
			println(labelMap.get(k).toString)
			println(singleClassTweets.size.toString)
			
			val shuffled = Random.shuffle(singleClassTweets)
			val (trainTweetR, testTweetR) = shuffled.splitAt(shuffled.size/2)
			trainTweetsB ++= trainTweetR
			testTweetsB ++= testTweetR
//			printf(trainTweetsB)
		}

		val trainTweets = trainTweetsB.toArray
		val testTweets = testTweetsB.toArray
		println(trainTweets.size.toString)
		println(testTweets.size.toString)

    DataStatistics(trainTweets, testTweets)

    val trainTweetsRDD = sc.parallelize(trainTweets, training_partitions)

    val (word2VecModel, logisticRegressionModel, _) = PerformTraining(sc, trainTweetsRDD)
		val testTweetsRDD = sc.parallelize(testTweets, testing_partitions)

    PerformPrediction(sc, word2VecModel, logisticRegressionModel, testTweetsRDD)

  }

	//based on above traintweetmodels which works. Just hack in the website into Tweet data format.
	def TrainWebsiteModelsBasedTweet(trainTableName:String, webTrain:String, webTest:String, sc:SparkContext):Unit = {
    println("Training website models using tweet methodology")

		//load training files... labelMap is for string to double mapping which is weirdish
    var labelMap = scala.collection.mutable.Map[String,Double]()
    val training_partitions = 8
    val testing_partitions = 8
    println("Training models")


		//randomly pick out tweets for testing and training
    //load website data from .csv provided by CMW team.
    val webpages = getWebpagesFromTable(trainTableName,webTrain, sc).collect()

		println("*********")
		var trainWebsB = ArrayBuffer[Tweet]()
		var testWebsB = ArrayBuffer[Tweet]()
		for((k,v) <- labelMap){
			val singleClassWebs = allTweets.filter(y => y.label == labelMap.get(k))
			println(labelMap.get(k).toString)
			println(singleClassWebs.size.toString)
			
			val shuffled = Random.shuffle(singleClassWebs)
			val (trainWebsR, testWebsR) = shuffled.splitAt(shuffled.size/2)
			trainWebsB ++= trainWebsR
			testWebsB ++= testWebsR
		}

		val trainWebs = trainWebsB.toArray
		val testWebs = testWebsB.toArray
		println(trainWebs.size.toString)
		println(testWebs.size.toString)

    DataStatistics(trainWebs, testWebs)

    val trainWebsRDD = sc.parallelize(trainWebs, training_partitions)
    val (word2VecModel, logisticRegressionModel, _) = PerformTraining(sc, trainWebsRDD)
		val testWebsRDD = sc.parallelize(testWebs, testing_partitions)

    PerformPrediction(sc, word2VecModel, logisticRegressionModel, testWebsRDD)
	}

///////////////////////////////////////
	//initial get website data from SMW .csv file. each file is its own topic already, so provide the desired label ID!
	def getWebsitesFromRawCsv(fileName:String, labelDouble: Double, sc: SparkContext): RDD[Tweet] = {
		//load file of rwa website data
    val file = sc.textFile(fileName)

		//map websites
    file.map(x => x.split(",", 5)).filter( y => (y.length == 5 && y(0) != "id")).map(x => Tweet(x(0),x(4), Option(labelDouble)))
  }

	def getWebpagesFromTable(tableName:String, fileName:String, sc: SparkContext): RDD[Tweet] = {
		//load file of rwa website data
    val file = sc.textFile(fileName)
		//map websites row keys into RDD[String]
    val rowKeys = file.map(x => x.split("\t", 23)).filter( y => (y.length == 23 && y(0) != "url-timestamp")).map(x => x(0))
		
		val conf = new HBaseConfiguration.create()
		val table = new HTable(conf, tableName)
		//load website clean text into RDD[Tweet]
		rowKeys.map( rowKey => Tweet(rowKey, Bytes.toString((table.get(new Get(Bytes.toBytes(rowKey)))).getValue(Bytes.toBytes("clean-webpage"),Bytes.toBytes("clean-text-profanity"))), Option(labelDouble)) )
  }


/////////////////////////////////////
  def getTweetsFromFile(fileName:String,labelMap:scala.collection.mutable.Map[String,Double], sc: SparkContext): RDD[Tweet] = {
    val file = sc.textFile(fileName)
    val allProductNum = file.map(x => x.split(",", 3)).filter( y => (y.length == 3 && y(0) != "id")).map(x => x(2)).distinct().collect()
    var maxLab = 0.0
		labelMap += ("0.0" -> 0.0)
    if (labelMap.nonEmpty ){
      maxLab = labelMap.valuesIterator.max + 1
    }
    allProductNum.foreach(num => {
      if (!labelMap.contains(num)){
        labelMap += (num -> maxLab)
        maxLab = maxLab + 1
      }
    })
    file.map(x => x.split(",", 4)).map(x => Tweet(x(0),x(1), labelMap.get(x(2))))
  }

   def PerformPrediction(sc: SparkContext, word2VecModel: Word2VecModel, logisticRegressionModel: LogisticRegressionModel, cleaned_testTweetsRDD: RDD[Tweet]) = {
    val teststart = System.currentTimeMillis()
    val (predictionTweets,predictionLabel) = Word2VecClassifier.predict(cleaned_testTweetsRDD, sc, word2VecModel, logisticRegressionModel)
    //val metricBasedPrediction = cleaned_testTweetsRDD.map(x => x.label.get).zip(predictions.map(x => x.label.get)).map(x => (x._2, x._1))
    Word2VecClassifier.GenerateClassifierMetrics(predictionLabel, "LRW2VClassifier", Word2VecClassifier._numberOfClasses)
    val testEnd = System.currentTimeMillis()
    println(s"Took ${(testEnd - teststart) / 1000.0} seconds for the prediction.")
  }
/////////////////////
  def PerformTraining(sc: SparkContext, cleaned_trainingTweetsRDD: RDD[Tweet]) = {
    val trainstart = System.currentTimeMillis()
    val (word2VecModel, logisticRegressionModel) = Word2VecClassifier.train(cleaned_trainingTweetsRDD, sc)
    val trainend = System.currentTimeMillis()
    println(s"Took ${(trainend - trainstart) / 1000.0} seconds for the training.")
    (word2VecModel, logisticRegressionModel, (trainend-trainstart)/1000.0)
  }
///////////////////////

  private def DataStatistics(trainTweets: Array[Tweet], testTweets: Array[Tweet]) = {
    //place a debug point or prints to see the statistics
    val trainCount = trainTweets.length
    val testCount = testTweets.length
    val bothTweets = trainTweets ++ testTweets
    val numClasses = bothTweets.map(x => x.label).distinct.length
    val minClass = bothTweets.groupBy(x => x.label).map(t => (t._1, t._2.length)).valuesIterator.min
    val minClassCount = bothTweets.groupBy(x => x.label).map(t => (t._1, t._2.length)).toList.count(x => x._2 == minClass)
    val maxClass = bothTweets.groupBy(x => x.label).map(t => (t._1, t._2.length)).valuesIterator.max
    val numToAmount = bothTweets.groupBy(x => x.label).map(t => (t._1, t._2.length)).toList.groupBy(x => x._2).mapValues(_.size)
    println("Histogram begin")
    for (i <- 1 to maxClass){
      println(i + "\t" + numToAmount.getOrElse(i,0))
    }
    println("the stats have been generated")
  }

  def PerformIDFTraining(sc: SparkContext, cleaned_trainingTweetsRDD: RDD[Tweet]) = {
    val trainstart = System.currentTimeMillis()
    val (idfModel, hashingModel, logisticRegressionModel) = Word2VecClassifier.trainIdfClassifer(cleaned_trainingTweetsRDD, sc)
    val trainend = System.currentTimeMillis()
    println(s"Took ${(trainend - trainstart) / 1000.0} seconds for the IDF model training.")
    (idfModel, hashingModel, logisticRegressionModel, (trainend - trainstart) / 1000.0)
  }
}

//
