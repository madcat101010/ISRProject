package isr.project

import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{HTable, Result, Scan}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
/**
  * Created by Eric on 11/8/2016.
  */
case class Tweet(id: String, tweetText: String, label: Option[Double] = None)
object DataRetriever {
  val _cachedRecordCount = 5000
  var _tableName: String = "ideal-cs5604f16" /*"ideal-cs5604f16-fake"*/
  var _colFam : String = "tweet"
  var _col : String = "cleantext" /*"text"*/

  def retrieveTweets(collectionID: String, sc : SparkContext): RDD[Tweet] = {
    //implicit val config = HBaseConfig()
    val scan = new Scan(Bytes.toBytes(collectionID), Bytes.toBytes(collectionID + '0'))
    val hbaseConf = HBaseConfiguration.create()
    val table = new HTable(hbaseConf,_tableName)
    scan.addColumn(Bytes.toBytes(_colFam), Bytes.toBytes(_col))
    scan.setCaching(_cachedRecordCount)
    scan.setBatch(1)
    val resultScanner = table.getScanner(scan)
    println(s"Caching Info:${scan.getCaching} Batch Info: ${scan.getBatch}")
    println("Scanning results now.")
    var continueLoop = true
    var totalRecordCount = 0
    import unicredit.spark.hbase._
    implicit val config = HBaseConfig(hbaseConf)
    try {
      val cols = Map(
      _colFam -> Set(_col)
    )
      val rdd = sc.hbase[String](_tableName,cols,scan)
      rdd.take(5)
      rdd.collect()
//      sc.parallelize(rdd.map(r => r._2.flatMap(n => n._2))
//      val results = resultScanner.next()
//      //var resultTweets = rowToTweetConverter(resultScanner.iterator())
//      //val rdd = sc.parallelize(resultScanner).map(r => rowToTweetConverter(r))
//      println("*********** Cleaning the tweets now. *****************")
//      val cleanTweets = CleanTweet.clean(rdd, sc)
//      println("*********** Predicting the tweets now. *****************")
//      val predictedTweets = Word2VecClassifier.predict(cleanTweets, sc)
//      println("*********** Persisting the tweets now. *****************")
//      DataWriter.writeTweets(predictedTweets)
//      if (results == null)
//        continueLoop = false
//      else {
//        totalRecordCount = totalRecordCount + 1
//        println(results)
//      }
    }

      catch {
        case e: Exception =>
          println(e.printStackTrace())
          println("Exception Encountered")
          println(e.getMessage)
      }



    println(s"Total record count:${totalRecordCount}")
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
    val cell = result.getColumnLatestCell(Bytes.toBytes(_colFam), Bytes.toBytes(_col))
    val key = Bytes.toString(cell.getRowArray, cell.getRowOffset, cell.getRowLength)
    val words = Bytes.toString(cell.getValueArray, cell.getValueOffset, cell.getValueLength)
    Tweet(key,words)
  }

}

