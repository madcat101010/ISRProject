/*


*/
package isr.project
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{HTable, Put}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.rdd.RDD

object DataWriter {


	Map labelIdMap = Map(0.0 -> "CouldNotClassify");


  def writeTweets(tweetRDD: RDD[Tweet], _tableName:String): Unit = {
		val _colFam = DataRetriever._classificationColFam;
		val _col = DataRetriever._classCol;
    val interactor = new HBaseInteraction(_tableName);
		collectedTweet = tweetRDD.collect();
    collectedTweet.foreach(tweet => interactor.putValueAt(_colFam, _col, tweet.id, labelMapper(tweet.label.getOrElse(0.0))))
    interactor.close()
 }


  def writeTrainingData(tweets: RDD[Tweet]): Unit = {
    val _tableName: String = "cs5604-f16-cla-training"
    val _textColFam: String = "training-tweet"
    val _labelCol: String = "label"
    val _textCol : String = "text"
    tweets.foreachPartition(
      tweetRDD => {
        val hbaseConf = HBaseConfiguration.create()
        val table = new HTable(hbaseConf,_tableName)
        tweetRDD.map(tweet => {
          val put = new Put(Bytes.toBytes(tweet.id))
          put.addColumn(Bytes.toBytes(_textColFam), Bytes.toBytes(_labelCol), Bytes.toBytes(tweet.label.get.toString))
          put.addColumn(Bytes.toBytes(_textColFam), Bytes.toBytes(_textCol), Bytes.toBytes(tweet.tweetText))
          put
        }).foreach(table.put)
      }
    )

  }

  def writeTweetToDatabase(tweet: Tweet, colFam: String, col: String, table: HTable): Put = {
    val putAction = putValueAt(colFam, col, tweet.id, labelMapper(tweet.label.getOrElse(0.0)), table)
    putAction
  }

  def putValueAt(columnFamily: String, column: String, rowKey: String, value: String, table: HTable) : Put = {
    // Make a new put object to handle adding data to the table
    // https://hbase.apache.org/apidocs/org/apache/hadoop/hbase/client/Put.html
    val put = new Put(Bytes.toBytes(rowKey))

    // add data to the put
    put.add(Bytes.toBytes(columnFamily), Bytes.toBytes(column), Bytes.toBytes(value))

    // put the data in the table
    put
  }

	def mapLabel(labelId:doudble, label:String){
		labalIdMap = labelIdMap + (labelId -> label);
	}
	
  def labelMapper(label:Double) : String= {
    labelIdMap.getOrElse(label,"ClassLabelMapError");
  }
}

//
