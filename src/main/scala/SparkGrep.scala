/*


*/

package isr.project

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Word2VecClassifier
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.HBaseAdmin
import org.apache.hadoop.hbase.client.{ConnectionFactory, HTable, Result, Scan, Get}
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
		if(args.length >= 8){
			//ensure correct usage
			if(args(0) != "train" && args(0) != "classify" && args(0) != "label"){
				System.err.println("Usage Error: args(0) != 'train' or 'classify' or 'label'");
      	System.exit(1);
			}
			if(args(1) != "webpage" && args(1) != "tweet" && args(1) != "w2v"){
				System.err.println("Usage Error: args(1) != 'tweet' or 'webpage' or 'w2v'");
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
				if(args(1) == "w2v"){
					val w2vModel = loadGoogleW2VBin("./data/GoogleNews-vectors-negative300.bin.gz")
					w2vModel.save(sc, "./data/training/GoogleNews-vectors-negative300_w2v.model")
				}
				else if(args(1) == "tweet"){
					var tweetTrainingFile = ("./data/training/" + eventName + "_tweet_training.csv");	//TODO:ensure file ending is good
					Word2VecClassifier._lrModelFilename = "./data/" + eventName + "_tweet_lr.model";
					//Word2VecClassifier._word2VecModelFilename = "./data/" + eventName + "_tweet_w2v.model";
					Word2VecClassifier._word2VecModelFilename = "./data/training/GoogleNews-vectors-negative300_w2v.model";
					TrainTweetModels(tweetTrainingFile, Word2VecClassifier._word2VecModelFilename, sc);	//TODO: File formatting must match when we make the training/testing files...
				}
				else if(args(1) == "webpage"){
					var websiteTrainingFile = ("./data/training/" + eventName + "_webpage_training.csv");	//TODO:ensure file ending is good
					Word2VecClassifier._lrModelFilename = "./data/" + eventName + "_webpage_lr.model";
					//Word2VecClassifier._word2VecModelFilename = "./data/" + eventName + "_webpage_w2v.model";
					Word2VecClassifier._word2VecModelFilename = "./data/training/GoogleNews-vectors-negative300_w2v.model";
					TrainWebpageModelsBasedTweet(tableNameSrc,websiteTrainingFile, Word2VecClassifier._word2VecModelFilename, sc)  //TODO: Don't combine csv file anymore... don't random pick train:test data
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
		    	DataRetriever.retrieveTweets(eventName, collectionName, 20000, tableNameSrc, tableNameDest, sc)
				}
				else if(args(1) == "webpage"){
		    	DataRetriever.retrieveWebpages(eventName, collectionName, 1000, tableNameSrc, tableNameDest, sc)
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
					val trainingTweets = DataRetriever.getTrainingTweets(sc, args(2), args(5), 500)
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
  
  def TrainTweetModels(trainFile: String, w2vModelFile: String, sc: SparkContext): Unit = {
    println("Training Tweet models")

		//load training files
    var labelMap = scala.collection.mutable.Map[String,Double]()
    val training_partitions = 8
    val testing_partitions = 8


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
			val (trainTweetR, testTweetR) = shuffled.splitAt(shuffled.size*(0.8))
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
		val word2vecModel = Word2VecModel.load(sc, w2vModelFile)			
    val logisticRegressionModel = PerformTraining(sc, trainTweetsRDD, word2vecModel)
		val testTweetsRDD = sc.parallelize(testTweets, testing_partitions)

    PerformPrediction(sc, word2vecModel, logisticRegressionModel, testTweetsRDD)

  }

	//based on above traintweetmodels which works. Just hack in the website into Tweet data format.
	def TrainWebpageModelsBasedTweet(trainTableName:String, webTrain:String, w2vFileName:String, sc:SparkContext):Unit = {
    println("Training website models using tweet methodology")

		//load training files... labelMap is for string to double mapping which is weirdish
    var labelMap = scala.collection.mutable.Map[String,Double]()
    val training_partitions = 8
    val testing_partitions = 8
    println("Training models")


		//randomly pick out tweets for testing and training
    //load website data from .csv provided by CMW team.
    val webpages = getWebpagesFromTable(trainTableName,webTrain,labelMap, sc).collect().toBuffer

		println("*********")
		var trainWebsB = ArrayBuffer[Tweet]()
		var testWebsB = ArrayBuffer[Tweet]()
		for((k,v) <- labelMap){
			val singleClassWebs = webpages.filter(y => y.label == labelMap.get(k))
			println(labelMap.get(k).toString)
			println(singleClassWebs.size.toString)
			
			val shuffled = Random.shuffle(singleClassWebs)
			val (trainWebsR, testWebsR) = shuffled.splitAt(shuffled.size*(0.8))
			trainWebsB ++= trainWebsR
			testWebsB ++= testWebsR
		}

		val trainWebs = trainWebsB.toArray
		val testWebs = testWebsB.toArray
		println(trainWebs.size.toString)
		println(testWebs.size.toString)

    DataStatistics(trainWebs, testWebs)

    val trainWebsRDD = sc.parallelize(trainWebs, training_partitions)
		val word2vecModel = Word2VecModel.load(sc, w2vModelFile)
    val logisticRegressionModel = PerformTraining(sc, trainWebsRDD, word2vecModel)
		val testWebsRDD = sc.parallelize(testWebs, testing_partitions)

    PerformPrediction(sc, word2vecModel, logisticRegressionModel, testWebsRDD)
	}

///////////////////////////////////////
	//initial get website data from SMW .csv file. each file is its own topic already, so provide the desired label ID!
	def getWebpagesFromRawCsv(fileName:String, labelDouble: Double, sc: SparkContext): RDD[Tweet] = {
		//load file of rwa website data
    val file = sc.textFile(fileName)

		//map websites
    file.map(x => x.split(",", 5)).filter( y => (y.length == 5 && y(0) != "id")).map(x => Tweet(x(0),x(4), Option(labelDouble)))
  }

	def getWebpagesFromTable(tableName:String, fileName:String,labelMap:scala.collection.mutable.Map[String,Double], sc: SparkContext): RDD[Tweet] = {
		//load file of rwa website data
    val file = sc.textFile(fileName)
		//map websites row keys into RDD[(String,Double)]
    val rowKeys : RDD[(String, Double)] = file.map(x => x.split("\t", 23)).filter( y => (y.length == 23 && y(22) != "spam label" && y(22) != "")).map(x => ( x(0), x(22).toDouble+1.0) )
		
		val allProductNum = file.map(x => x.split("\t", 23)).filter( y => (y.length == 23 && y(22) != "spam label" && y(22) != "")).map(x => (x(22).toDouble+1.0).toString).distinct().collect()
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

		val rowKeySer = rowKeys.collect()
		val conf = HBaseConfiguration.create()
		val table = new HTable(conf, tableName)
		//load website data from table cannot be parallized via RDD, can only be serial. Hence use a map and then parallize to RDD
		var dataMap = scala.collection.mutable.Map[String,(String, String)]()
		for( (rowkey, label) <- rowKeySer){
			//println(rowkey)
			if(rowkey != ""){
				val resultRow = table.get(new Get(Bytes.toBytes(rowkey)))
				if (!resultRow.isEmpty()){
					val getData = resultRow.getValue(Bytes.toBytes("clean-webpage"),Bytes.toBytes("clean-text-profanity"))
					if (getData != null){
						dataMap += (rowkey -> (Bytes.toString(getData), label.toString) )
						println("ADDED:" + rowkey + " | " + label.toString)
					}
				}
			}
		}
		if(!dataMap.isEmpty){
			val dataMapRDD = sc.parallelize(dataMap.toSeq)
			val rowText : RDD[(String, String, String)] = dataMapRDD.map( row => ( row._1, row._2._1, row._2._2 ) )
			rowText.map( row => Tweet(row._1, row._2 , labelMap.get(row._3)) )
		}
		else{
			println("ERROR: Empty dataName in getWebpagesFromTable")
			null
		}
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
  def PerformTraining(sc: SparkContext, cleaned_trainingTweetsRDD: RDD[Tweet], w2vModel:Word2VecModel) = {
    val trainstart = System.currentTimeMillis()
    val logisticRegressionModel = Word2VecClassifier.train(cleaned_trainingTweetsRDD, Word2VecModel, sc)
    val trainend = System.currentTimeMillis()
    println(s"Took ${(trainend - trainstart) / 1000.0} seconds for the training.")
    (logisticRegressionModel, (trainend-trainstart)/1000.0)
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


	//direct read the .bin.gz file
	private def loadGoogleW2VBin(file: String) : Word2VecModel = {
		def readUntil(inputStream: DataInputStream, term: Char, maxLength: Int = 1024 * 8): String = {
		  var char: Char = inputStream.readByte().toChar
		  val str = new StringBuilder
		  while (!char.equals(term)) {
		    str.append(char)
		    assert(str.size < maxLength)
		    char = inputStream.readByte().toChar
		  }
		  str.toString
		}
		val inputStream: DataInputStream = new DataInputStream(new GZIPInputStream(new FileInputStream(file)))
		try {
		  val header = readUntil(inputStream, '\n')
		  val (records, dimensions) = header.split(" ") match {
		    case Array(records, dimensions) => (records.toInt, dimensions.toInt)
		  }
		  val w2vModel = new Word2VecModel((0 until records).toArray.map(recordIndex => {
		    readUntil(inputStream, ' ') -> (0 until dimensions).map(dimensionIndex => {
		      java.lang.Float.intBitsToFloat(java.lang.Integer.reverseBytes(inputStream.readInt()))
		    }).toArray
		  }).toMap)
			return w2vModel
		} finally {
		  inputStream.close()
		}
	}




}


//
