/*
This is an adaptation of the original DataRetriever.scala framework for website classification.
TODOs mark where we can put our own cleaning code and other models we want. If we add other models for testing purposes,
we will need to add the the sparkcontext broadcast for those along with the trained model file locations.

*/

package isr.project


import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{ConnectionFactory, HTable, Result, Scan}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Word2VecClassifier

import scala.collection.JavaConversions._

case class Website(id: String, cleanText: String, label: Option[Double] = None)

object ClassifyWebsite {
  val _websiteModelURL = "data/website/"
  var _webTableName: String = "WebsiteTable" /*the table which new websites come in from*/
  var _columnFamily : String = "website"
  var _Column : String = "cleanText" 	/*the cleaned website html text column*/

  def classifyWebsite (classificationName: String, collectionID: String, sc: SparkContext, cacheRecCount: Int = 50): RDD[Website] = {
    //Make the file URL to the models we need
		val bcWebsiteModel_lr = _websiteModelURL + classificationName + "_lr.model"
		val bcWebsiteModel_w2v = _websiteModelURL + classificationName + "_w2v.model"

		//Tell all nodes the file via broadcast
		val bcModelFilename_w2v = sc.broadcast(bcWebsiteModel_lr)
    val bcClassifierModelFilename_lr = sc.broadcast(bcWebsiteModel_w2v)

		//Load the broadcasted model files
		val word2vecModel = Word2VecModel.load(sc, bcModelFilename_w2v.value)
    val logisticRegressionModel = LogisticRegressionModel.load(sc, bcClassifierModelFilename_lr.value)
    println(s"W2V Model file found:$bcModelFilename_w2v. Loading model.")
    println(s"LR Model file found:$bcClassifierModelFilename_lr. Loading model.")

    //Perform a cold start of the model pipeline so that this loading doesn't disrupt the read later.
    val coldWebsite = sc.parallelize(Array[Website]{ Website("id", "Some tweet")})
    val (predictedWebsite,_) = Word2VecClassifier.predict(coldWebsite, sc, word2vecModel, logisticRegressionModel)
    predictedWebsite.count

    // scan over only the collection
    val scan = new Scan(Bytes.toBytes(collectionID), Bytes.toBytes(collectionID + '0'))
    val hbaseConf = HBaseConfiguration.create()
    val table = new HTable(hbaseConf,_tableName)

    // add the specific column to scan
    scan.addColumn(Bytes.toBytes(_columnFamily), Bytes.toBytes(_Column))

    // add caching to increase speed
    scan.setCaching(cacheRecCount)
    scan.setBatch(100)
    val resultScanner = table.getScanner(scan)
    println(s"Caching Info:${scan.getCaching} Batch Info: ${scan.getBatch}")
    println("Scanning results now.")

    var continueLoop = true
    var totalRecordCount: Long = 0
    while (continueLoop) {
      try {
        println("Getting next batch of results now.")
        val start = System.currentTimeMillis()

        val results = resultScanner.next(cacheRecCount)

        if (results == null || results.isEmpty)
          continueLoop = false
        else {
          println(s"Result Length:${results.length}")

					//read website data from hbase to local map for processing
          val resultWebsite = results.map(r => rowToWebsiteConverter(r))
          val rddW = sc.parallelize(resultWebsite)
          rddW.cache()
          rddW.repartition(12)

          println("*********** Cleaning the website data now. *****************")
          //val cleanWebsites = CleanWebsites.clean(rddW, sc) //TODO: IF we need to clean website stuff more, put it here...
          
					println("*********** Predicting the website data now. *****************")
					//TODO: Run model code here...
          val (predictedWebsites,_) = Word2VecClassifier.predict(cleanWebsites, sc, word2vecModel, logisticRegressionModel)

          println("*********** Persisting the websites now. *****************")
          val repartitionedPredictions = predictedWEbsites.repartition(12)
          DataWriter.writeTweets(repartitionedPredictions)

          predictedTweets.cache()
          val batchTweetCount = predictedTweets.count()
          println(s"The amount of tweets to be written is $batchTweetCount")
          val end = System.currentTimeMillis()
          totalRecordCount += batchTweetCount
          println(s"Took ${(end-start)/1000.0} seconds for This Batch.")
          println(s"This batch had $batchTweetCount tweets. We have processed $totalRecordCount tweets overall")
        }
      }
      catch {
        case e: Exception =>
          println(e.printStackTrace())
          println("Exception Encountered")
          println(e.getMessage)
          continueLoop = false
      }

    }

    println(s"Total record count:$totalRecordCount")
    resultScanner.close()
    //val interactor = new HBaseInteraction(_tableName)

    return null



    /*val scanner = new Scan(Bytes.toBytes(collectionID), Bytes.toBytes(collectionID + '0'))
    val cols = Map(
      _colFam -> Set(_col)
    )*/
    //val rdd = sc.hbase[String](_tableName,cols,scanner)
    //val result  = interactor.getRowsBetweenPrefix(collectionID, _colFam, _col)
    //sc.parallelize(result.iterator().map(r => rowToTweetConverter(r)).toList)
    //rdd.map(v => Tweet(v._1, v._2.getOrElse(_colFam, Map()).getOrElse(_col, ""))).foreach(println)
    //rdd.map(v => Tweet(v._1, v._2.getOrElse(_colFam, Map()).getOrElse(_col, "")))/*.repartition(sc.defaultParallelism)*/.filter(tweet => tweet.tweetText.trim.isEmpty)
  }

  def rowToTweetConverter(result : Result): Tweet ={
    val cell = result.getColumnLatestCell(Bytes.toBytes(_columnFamily), Bytes.toBytes(_Column))
    val key = Bytes.toString(cell.getRowArray, cell.getRowOffset, cell.getRowLength)
    val words = Bytes.toString(cell.getValueArray, cell.getValueOffset, cell.getValueLength)
    Tweet(key,words)
    null
  }

  def retrieveTrainingTweetsFromFile(fileName:String, sc : SparkContext) : RDD[Tweet] = {
    val lines = sc.textFile(fileName)
    lines.map(line=> Tweet(line.split('|')(1), line.split('|')(2), Option(line.split('|')(0).toDouble))).filter(tweet => tweet.label.isDefined)
  }

  def getTrainingTweets(sc:SparkContext): RDD[Tweet] = {
    val _tableName: String = "cs5604-f16-cla-training"
    val _textColFam: String = "training-tweet"
    val _labelCol: String = "label"
    val _textCol : String = "text"
    val connection = ConnectionFactory.createConnection()
    val table = connection.getTable(TableName.valueOf(_tableName))
    val scanner = new Scan()
    scanner.addColumn(Bytes.toBytes(_textColFam), Bytes.toBytes(_labelCol))
    scanner.addColumn(Bytes.toBytes(_textColFam), Bytes.toBytes(_textCol))
    sc.parallelize(table.getScanner(scanner).map(result => {
      val labcell = result.getColumnLatestCell(Bytes.toBytes(_textColFam), Bytes.toBytes(_labelCol))
      val textcell = result.getColumnLatestCell(Bytes.toBytes(_textColFam), Bytes.toBytes(_textCol))
      val key = Bytes.toString(labcell.getRowArray, labcell.getRowOffset, labcell.getRowLength)
      val words = Bytes.toString(textcell.getValueArray, textcell.getValueOffset, textcell.getValueLength)
      val label = Bytes.toString(labcell.getValueArray, labcell.getValueOffset, labcell.getValueLength).toDouble
      Tweet(key,words,Option(label))
    }).toList)
  }



}

