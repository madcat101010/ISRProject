/*


*/

package isr.project

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Word2VecClassifier
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SparkGrep {
  def tweetchange(tweet: Tweet): Tweet = {
    if (tweet.label.get == 0.0) {
      return Tweet(tweet.id, tweet.tweetText, Option(9.0))
    }
    return Tweet(tweet.id, tweet.tweetText, tweet.label)
  }

//SparkGrep <train/classify> <website/tweet> [<YAML Config File>/collection name] [class1Name] [class2Name] [...]
  def main(args: Array[String]) {
		if(args.length >= 3){
			//ensure correct usage
			if(args(0) != "train" || args(0) != "classify"){
				System.err.println("Usage Error: args(0) != 'train' or 'classify'");
      	System.exit(1);
			}
			if(args(1) != "website" || args(1) != "tweet"){
				System.err.println("Usage Error: args(1) != 'tweet' or 'website'");
      	System.exit(1);
			}
			//if(args(2)){

			//}

			//load collection name and class mapping for that collection
			val collectionName = args(2);
			var x = 3;
			for( x <- 3 to (args.length-1) ){
				DataWriter.mapLabel( (x-3).toDouble, args(x) );
			}
			Word2VecClassifier._numberOfClasses = (x-2); 
			var tableName = "getar-cs5604f17-eclipse-collection_shard1_replica1";

			//train or classify
			if(args(0) == "train"){
				val conf = new SparkConf()
				    .setMaster("local[*]")
				    .setAppName("CLA-ModelTraining")
				val sc = new SparkContext(conf);
				if(args(1) == "tweet"){
					var tweetTrainingFile = ("./data/training/" + collectionName + "_tweet_training.csv");	//TODO:ensure file ending is good
					var tweetTestingFile = ("./data/testing/" + collectionName + "_tweet_testing.csv");	//TODO:ensure file ending is good
					Word2VecClassifier._lrModelFilename = "./data/" + collectionName + "_tweet_lr.model";
					Word2VecClassifier._word2VecModelFilename = "./data/" + collectionName + "_tweet_w2v.model";
				  TrainTweetModels(tweetTrainingFile, tweetTestingFile, sc);	//TODO: File formatting must match when we make the training/testing files...
				}
				else if(args(1) == "website"){
					var websiteTrainingFile = ("./data/training/" + collectionName + "_website_training.csv");	//TODO:ensure file ending is good
					var websiteTestingFile = ("./data/testing/" + collectionName + "_website_testing.csv");	
					//TrainWebsiteModelsBasedTweet("./data/website_shooting_data/dhs_shooting.csv", "./data/website_shooting_data/niu_shooting.csv", sc)  //TODO: Don't combine csv file anymore... don't random pick train:test data
				}
				System.exit(0)
			}
			else if(args(0) == "classify"){
		    Logger.getLogger("org").setLevel(Level.OFF)
    		Logger.getLogger("akka").setLevel(Level.OFF)
    		val start = System.currentTimeMillis()
				val conf = new SparkConf()
		        .setMaster("local[*]")
		        .setAppName("CLA-Classifying")
		    val sc = new SparkContext(conf)

				if(args(1) == "tweet"){
		    	val readTweets = DataRetriever.retrieveTweets(collectionName, 100, tableName, sc)
				}
				else if(args(1) == "website"){
					println("TODO: classify website");
				}
    		val end = System.currentTimeMillis()
    		println("Took ${(end - start) / 1000.0} seconds for the whole process.")
				System.exit(0)
			}
		}
    else{
      System.err.println("Usage: SparkGrep <train/classify> <website/tweet> <YAML Config File>")
			System.err.println("Usage Note: files should be in the ./data/ directory i.e. 'data/collection1_trainingfile'")
      System.exit(1)
    }

  }

  def TrainTweetModels(trainFile: String, testFile: String, sc: SparkContext): Unit = {
    println("Training models")

		//load training files
    var labelMap = scala.collection.mutable.Map[String,Double]()
    val training_partitions = 8
    val testing_partitions = 8
    val trainTweets = getTweetsFromFile(trainFile, labelMap, sc).collect()
    val testTweets = getTweetsFromFile(testFile, labelMap, sc).collect()
		
    DataStatistics(trainTweets, testTweets)
    SetupWord2VecField(trainFile, getTweetsFromFile(trainFile, labelMap, sc))

    val trainTweetsRDD = sc.parallelize(trainTweets, training_partitions)

		//test that RDD map has the expected text file data
		//println("Get here")    
		//println("#############################################################")
		//for ((k,v) <- trainTweets) printf("key: %s, value: %s", k, v)
		//println(trainTweets.mkString(" "))
		//println(trainTweetsRDD)
		
		//val cleaned_trainingTweetsRDD = sc.parallelize(CleanTweet.clean(trainTweetsRDD,sc).collect(),training_partitions).cache()

    val (word2VecModel, logisticRegressionModel, _) = PerformTraining(sc, trainTweetsRDD)
		val testTweetsRDD = sc.parallelize(testTweets, testing_partitions)
    //val cleaned_testTweetsRDD = sc.parallelize(CleanTweet.clean(testTweetsRDD,sc).collect(),testing_partitions).cache()

    PerformPrediction(sc, word2VecModel, logisticRegressionModel, testTweetsRDD)

  }

	//based on above traintweetmodels which works. Just hack in the website into Tweet data format.
	def TrainWebsiteModelsBasedTweet( web1:String, web2:String, sc:SparkContext):Unit = {
    println("Training website models using tweet methodology")

		//load training files... labelMap is for string to double mapping which is weirdish
    var labelMap = scala.collection.mutable.Map[String,Double]()
    val training_partitions = 8
    val testing_partitions = 8

    //load website data from .csv provided by CMW team.
    val websites_1 = getWebsitesFromRawCsv(web1, 0.0, sc).collect() //DunbarHighSchoolShooting
    val websites_2 = getWebsitesFromRawCsv(web2, 1.0, sc).collect() //NorthIllinoisUniversityShooting
		//combine odds of both website sets for training, and evens of both website sets for testing
		//websites_1.foreach(println)
		//websites_2.foreach(println)
		val web1_even = (websites_1.filter(f => f.id.toInt % 2 == 0))
		val web1_odd = (websites_1.filter(f => f.id.toInt % 2 != 0))
		val web2_even = (websites_2.filter(f => f.id.toInt % 2 == 0))
		val web2_odd = (websites_2.filter(f => f.id.toInt % 2 != 0))
		val trainTweets = web1_odd.union(web2_odd)
		val testTweets = web1_even.union(web2_even)
		
		//println("$$$$$$$$$TRAIN TWEETS$$$$$")
		//trainTweets.foreach(println)
		//println("######TEST TWEEETS####")
		//testTweets.foreach(println)

    DataStatistics(trainTweets, testTweets)
    SetupWord2VecField("website_tweethack_", sc.parallelize(trainTweets))

    val trainTweetsRDD = sc.parallelize(trainTweets, training_partitions)

		//test that RDD map has the expected text file data
		//println("Get here")    
		//println("#############################################################")
		//for ((k,v) <- trainTweets) printf("key: %s, value: %s", k, v)
		//println(trainTweets.mkString(" "))
		//println(trainTweetsRDD)
		
		//val cleaned_trainingTweetsRDD = sc.parallelize(CleanTweet.clean(trainTweetsRDD,sc).collect(),training_partitions).cache()

    val (word2VecModel, logisticRegressionModel, _) = PerformTraining(sc, trainTweetsRDD)
		val testTweetsRDD = sc.parallelize(testTweets, testing_partitions)
    //val cleaned_testTweetsRDD = sc.parallelize(CleanTweet.clean(testTweetsRDD,sc).collect(),testing_partitions).cache()

    PerformPrediction(sc, word2VecModel, logisticRegressionModel, testTweetsRDD)
	}

///////////////////////////////////////
	//initial get website data from SMW .csv file. each file is its own topic already, so provide the desired label ID!
	def getWebsitesFromRawCsv(fileName:String, labelDouble: Double, sc: SparkContext): RDD[Tweet] = {
		//load file of rwa website data
    val file = sc.textFile(fileName)

		//map websites
    file.map(x => x.split(",", 5)).filter( y => (y.length == 5 && y(0) != "id")).map(x => Tweet(x(0),x(4), Option(labelDouble)))
  }



/////////////////////////////////////
  def getTweetsFromFile(fileName:String,labelMap:scala.collection.mutable.Map[String,Double], sc: SparkContext): RDD[Tweet] = {
    val file = sc.textFile(fileName)
    val allProductNum = file.map(x => x.split(", ")).filter(_.length == 3).map(x => x(0)).distinct().collect() ++
      file.map(x => x.split('|')).filter(_.length == 2).map(x => x(0)).collect()
    var maxLab = 0.0
    if (labelMap.nonEmpty ){
      maxLab = labelMap.valuesIterator.max + 1
    }
    allProductNum.foreach(num => {
      if (!labelMap.contains(num)){
        labelMap += (num -> maxLab)
        maxLab = maxLab + 1
      }
    })
    file.map(x => x.split(", ")).filter(_.length == 3).map(x => Tweet(x(1),x(2), labelMap.get(x(0)))).union(
        file.map(x => x.split('|')).filter(_.length == 2).zipWithUniqueId().map(x => Tweet(x._2.toString,x._1(1),labelMap.get(x._1(0)))))
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
  def SetupWord2VecField(trainFile: String, trainTweets: RDD[Tweet]): Unit = {
    Word2VecClassifier._lrModelFilename = trainFile + "lrModel"
    Word2VecClassifier._word2VecModelFilename = trainFile + "w2vModel"
    Word2VecClassifier._numberOfClasses =  trainTweets.map(x => x.label).distinct.count().toInt
  }

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
