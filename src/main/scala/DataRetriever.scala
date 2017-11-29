/*


*/
package isr.project


import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.filter.{FilterList, SingleColumnValueFilter}
//import org.apache.hadoop.hbase.CompareOperator
import org.apache.hadoop.hbase.filter.CompareFilter.CompareOp
import org.apache.hadoop.hbase.client.{ConnectionFactory, HTable, Result, Scan}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Word2VecClassifier

import scala.collection.JavaConversions._
/**
  * Created by Eric on 11/8/2016.
  */
case class Tweet(id: String, tweetText: String, label: Option[Double] = None) //ID is the hbase table row key!

object DataRetriever {

	val _metaDataColFam : String = "metadata"
	val _metaDataTypeCol : String = "doc-type"
	val _metaDataCollectionNameCol : String = "collection-name"

	val _tweetColFam : String = "tweet"
	val _tweetUniqueIdCol : String = "tweet-id"
	val _cleanTweetColFam : String = "clean-tweet"
	val _cleanTweetTextCol : String = "clean-text-cla"
	val _cleanTweetSnerOrg : String = "sner-organizations"
	val _cleanTweetSnerLoc : String = "sner-locations"
	val _cleanTweetSnerPeople : String = "sner-people"
	val _cleanTweetLongURL : String = "long-url"
	val _cleanTweetHashtags : String = "hashtags"

	val _classificationColFam = "classification"
	val _classCol : String = "classification-list"
	val _classProbCol : String = "probability-list"

	val _cleanWebpageColFam : String = "clean-webpage"
	val _cleanWebpageTextCol : String "clean-text-profanity"

  def retrieveTweets(eventName:String, collectionName:String, _cachedRecordCount:Int, tableNameSrc:String, tableNameDest:String, sc: SparkContext): RDD[Tweet] = {
    //implicit val config = HBaseConfig()
		var _lrModelFilename = "./data/" + eventName + "_tweet_lr.model";
		var _word2VecModelFilename = "./data/" + eventName + "_tweet_w2v.model";
		

    val bcWord2VecModelFilename = sc.broadcast(_word2VecModelFilename)
    val bcLRClassifierModelFilename = sc.broadcast(_lrModelFilename)
    val word2vecModel = Word2VecModel.load(sc, bcWord2VecModelFilename.value)
    val logisticRegressionModel = LogisticRegressionModel.load(sc, bcLRClassifierModelFilename.value)
    println(s"Classifier Model file found:$bcLRClassifierModelFilename. Loading model.")
    //Perform a cold start of the model pipeline so that this loading
    //doesn't disrupt the read later.
    val coldTweet = sc.parallelize(Array[Tweet]{ Tweet("id", "Some tweet")})
    val predictedTweets = Word2VecClassifier.predictClass(coldTweet, sc, word2vecModel, logisticRegressionModel)
    predictedTweets.count

    // scan over only the collection
    val scan = new Scan()

    val hbaseConf = HBaseConfiguration.create()
    val srcTable = new HTable(hbaseConf, tableNameSrc)
		val destTable = new HTable(hbaseConf, tableNameDest)

		if( !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_metaDataColFam)) || !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_tweetColFam)) ){
			System.err.println("ERROR: Source tweet table missing required column family!");
			return null;
		}

		if( !destTable.getTableDescriptor().hasFamily(Bytes.toBytes(_classificationColFam)) ){
			System.err.println("ERROR: Destination tweet table missing required classification column family!");
			return null;
		}

	  	// MUST scan the column to filter using it... else it assumes column does not exist and will auto filter if setFilterIfMissing(true) is set.
		scan.addColumn(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataCollectionNameCol));
		scan.addColumn(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol));
		scan.addColumn(Bytes.toBytes(_tweetColFam), Bytes.toBytes(_tweetUniqueIdCol));
		scan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetTextCol));
		scan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetHashtags));
		scan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerOrg));
		scan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetLongURL));
		scan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerPeople));
		scan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerLoc));
	  	
		//will throw exception if table does not have classification family. If ommitted from scan, filter assumes unclassified and will keep the row
		if( srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_classificationColFam)) ){	
			scan.addColumn(Bytes.toBytes(_classificationColFam), Bytes.toBytes(_classCol));
		}

		//filter for only same collection, is tweet, has clean text, and not classified ... uncomment when table has the missing fields
		val filterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);

		println("Filter: Keeping Collection Name == " + collectionName)
		val filterCollect = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataCollectionNameCol), CompareOp.EQUAL , Bytes.toBytes(collectionName));
		filterCollect.setFilterIfMissing(true);	//filter all rows that do not have a collection name
		filterList.addFilter(filterCollect);

		println("Filter: Keeping Doc Type == tweet")
		val filterTweet = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol), CompareOp.EQUAL , Bytes.toBytes("tweet"));
		filterTweet.setFilterIfMissing(true);	//filter all rows that are not marked as tweets
		filterList.addFilter(filterTweet);

		println("Filter: Keeping Clean Text != ''")
		val filterNoClean = new SingleColumnValueFilter(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetTextCol), CompareOp.NOT_EQUAL , Bytes.toBytes(""));	//note compareOp vs compareOperator depending on hadoop version
		filterNoClean.setFilterIfMissing(true);	//filter all rows that do not have clean text column
		filterList.addFilter(filterNoClean);

		/*	Commented out for now so we can reclassify already classified data for now... consider input config arg	  	
		val filterUnclass = new SingleColumnValueFilter(Bytes.toBytes(_classificationColFam), Bytes.toBytes(_classCol), CompareOp.EQUAL , Bytes.toBytes(""));
		filterUnclass.setFilterIfMissing(false);	//keep only unclassified data
		filterList.addFilter(filterUnclass);
		*/
		
		scan.setFilter(filterList);


    // add caching to increase speed
    scan.setCaching(_cachedRecordCount)
    //scan.setBatch(-1)
    val resultScanner = srcTable.getScanner(scan)

    println(s"Caching Info:${scan.getCaching} Batch Info: ${scan.getBatch}")
    println("Scanning results now.")

    var continueLoop = true
    var totalRecordCount: Long = 0
    while (continueLoop) {
      try {
        println("Getting next batch of results now.")
        val start = System.currentTimeMillis()

        val results = resultScanner.next(_cachedRecordCount)

        if (results == null || results.isEmpty){
        	println("No more results from scan. Finishing...")
          continueLoop = false
				}
        else {
          println(s"Result Length:${results.length}")
          val resultTweets = results.map(r => rowToTweetConverter(r))
          val rddT = sc.parallelize(resultTweets)
          rddT.cache()
          rddT.repartition(12)
          //println("*********** Cleaning the tweets now. *****************")
          //val cleanTweets = CleanTweet.clean(rddT, sc)
          println("*********** Predicting the tweets now. *****************")
					val predictedTweets = Word2VecClassifier.predictClass(rddT, sc, word2vecModel, logisticRegressionModel)
          println("*********** Persisting the tweets now. *****************")

          val repartitionedPredictions = predictedTweets.repartition(12)
          DataWriter.writeTweets(repartitionedPredictions, tableNameDest)

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

  }


//

	def rowToTweetConverter(result : Result): Tweet ={
//		val _cleanTweetColFam : String = "clean-tweet"
//		val _cleanTweetTextCol : String = "clean-text-cla"
//		val _cleanTweetSnerOrg : String = "sner-organizations"
//		val _cleanTweetSnerLoc : String = "sner-locations"
//		val _cleanTweetSnerPeople : String = "sner-people"
//		val _cleanTweetLongURL : String = "long-url"
//		val _cleanTweetHashtags : String = "hashtags"
		val cell1 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetTextCol))
		val cell2 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetHashtags))
		val cell3 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerOrg))
		val cell4 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetLongURL))
		val cell5 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerPeople))
		val cell6 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerLoc))
			
		val key : String = Bytes.toString(result.getRow())
		var words : String = (Bytes.toString(cell1.getValueArray, cell1.getValueOffset, cell1.getValueLength) + " ");
		if(cell2 != null)
			words += (Bytes.toString(cell2.getValueArray, cell2.getValueOffset, cell2.getValueLength) + " ")
		if(cell3 != null)
			words += (Bytes.toString(cell3.getValueArray, cell3.getValueOffset, cell3.getValueLength) + " ")
		if(cell4 != null)
			words += (Bytes.toString(cell4.getValueArray, cell4.getValueOffset, cell4.getValueLength) + " ")
		if(cell5 != null)
			words += (Bytes.toString(cell5.getValueArray, cell5.getValueOffset, cell5.getValueLength) + " ")
		if(cell6 != null)
			words += (Bytes.toString(cell6.getValueArray, cell6.getValueOffset, cell6.getValueLength) + " ")
		words.dropRight(1)
		Tweet(key,words)
	}


  def retrieveTrainingTweetsFromFile(fileName:String, sc : SparkContext) : RDD[Tweet] = {
    val lines = sc.textFile(fileName)
    lines.map(line=> Tweet(line.split('|')(1), line.split('|')(2), Option(line.split('|')(0).toDouble))).filter(tweet => tweet.label.isDefined)
  }


//////////////////////////////////////////


	def retrieveWebpages(eventName:String, collectionName:String, _cachedRecordCount:Int, tableNameSrc:String, tableNameDest:String, sc: SparkContext): RDD[Tweet] = {
		//implicit val config = HBaseConfig()
		var _lrModelFilename = "./data/" + eventName + "_webpage_lr.model";
		var _word2VecModelFilename = "./data/" + eventName + "_webpage_w2v.model";


		val bcWord2VecModelFilename = sc.broadcast(_word2VecModelFilename)
		val bcLRClassifierModelFilename = sc.broadcast(_lrModelFilename)
		val word2vecModel = Word2VecModel.load(sc, bcWord2VecModelFilename.value)
		val logisticRegressionModel = LogisticRegressionModel.load(sc, bcLRClassifierModelFilename.value)
		println(s"Classifier Model file found:$bcLRClassifierModelFilename. Loading model.")
		//Perform a cold start of the model pipeline so that this loading
		//doesn't disrupt the read later.
		val coldTweet = sc.parallelize(Array[Tweet]{ Tweet("id", "Some tweet")})
		val predictedTweets = Word2VecClassifier.predictClass(coldTweet, sc, word2vecModel, logisticRegressionModel)
		predictedTweets.count

		// scan over only the collection
		val scan = new Scan()

		val hbaseConf = HBaseConfiguration.create()
		val srcTable = new HTable(hbaseConf, tableNameSrc)
		val destTable = new HTable(hbaseConf, tableNameDest)

		if( !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_metaDataColFam)) || !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_cleanWebpageColFam)) ){
			System.err.println("ERROR: Source webpage table missing required column family!");
			return null;
		}

		if( !destTable.getTableDescriptor().hasFamily(Bytes.toBytes(_classificationColFam)) ){
			System.err.println("ERROR: Destination webpage table missing required classification column family!");
			return null;
		}

	  	// MUST scan the column to filter using it... else it assumes column does not exist and will auto filter if setFilterIfMissing(true) is set.
		scan.addColumn(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataCollectionNameCol));
		scan.addColumn(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol));
		scan.addColumn(Bytes.toBytes(_cleanWebpageColFam), Bytes.toBytes(_cleanWebpageTextCol));

		//will throw exception if table does not have classification family. If ommitted from scan, filter assumes unclassified and will keep the row
		if( srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_classificationColFam)) ){	
			scan.addColumn(Bytes.toBytes(_classificationColFam), Bytes.toBytes(_classCol));
		}

		//filter for only same collection, is tweet, has clean text, and not classified ... uncomment when table has the missing fields
		val filterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);

		println("Filter: Keeping Collection Name == " + collectionName)
		val filterCollect = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataCollectionNameCol), CompareOp.EQUAL , Bytes.toBytes(collectionName));
		filterCollect.setFilterIfMissing(true);	//filter all rows that do not have a collection name
		filterList.addFilter(filterCollect);

		println("Filter: Keeping Doc Type == webpage")
		val filterTweet = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol), CompareOp.EQUAL , Bytes.toBytes("webpage"));
		filterTweet.setFilterIfMissing(true);	//filter all rows that are not marked as tweets
		filterList.addFilter(filterTweet);

		println("Filter: Keeping Clean Text != ''")
		val filterNoClean = new SingleColumnValueFilter(Bytes.toBytes(_cleanWebpageColFam), Bytes.toBytes(_cleanWebpageTextCol), CompareOp.NOT_EQUAL , Bytes.toBytes(""));	//note compareOp vs compareOperator depending on hadoop version
		filterNoClean.setFilterIfMissing(true);	//filter all rows that do not have clean text column
		filterList.addFilter(filterNoClean);
		
		/*	Commented out for now so we can reclassify already classified data for now... consider input config arg	  	
		val filterUnclass = new SingleColumnValueFilter(Bytes.toBytes(_classificationColFam), Bytes.toBytes(_classCol), CompareOp.EQUAL , Bytes.toBytes(""));
		filterUnclass.setFilterIfMissing(false);	//keep only unclassified data
		filterList.addFilter(filterUnclass);
		*/
		
		scan.setFilter(filterList);


		// add caching to increase speed
		scan.setCaching(_cachedRecordCount)
		val resultScanner = srcTable.getScanner(scan)

		println(s"Caching Info:${scan.getCaching} Batch Info: ${scan.getBatch}")
		println("Scanning results now.")

		var continueLoop = true
		var totalRecordCount: Long = 0
		while (continueLoop) {
			try {
				println("Getting next batch of results now.")
				val start = System.currentTimeMillis()

        val results = resultScanner.next(_cachedRecordCount)

        if (results == null || results.isEmpty){
					println("No more results from scan. Finishing...")
					continueLoop = false
				}
        else {
          println(s"Result Length:${results.length}")
          val resultTweets = results.map(r => rowToWebpageConverter(r))
          val rddT = sc.parallelize(resultTweets)
          rddT.cache()
          rddT.repartition(12)
          //println("*********** Cleaning the tweets now. *****************")
          //val cleanTweets = CleanTweet.clean(rddT, sc)
          println("*********** Predicting the tweets now. *****************")
					val predictedTweets = Word2VecClassifier.predictClass(rddT, sc, word2vecModel, logisticRegressionModel)
          println("*********** Persisting the tweets now. *****************")

          val repartitionedPredictions = predictedTweets.repartition(12)
          DataWriter.writeTweets(repartitionedPredictions, tableNameDest)	//only writes to classificaiton column family, and Tweet data struct has row key

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

  }

	
	def rowToWebpageConverter(result : Result): Tweet ={
		val cell1 = result.getColumnLatestCell(Bytes.toBytes(_cleanWebpageColFam), Bytes.toBytes(_cleanWebpageTextCol))
			
		val key : String = Bytes.toString(result.getRow())
		val words : String = Bytes.toString(cell1.getValueArray, cell1.getValueOffset, cell1.getValueLength)
		Tweet(key,words)
	}
	
	
	//////////////////////////////////


	def getTrainingTweets(sc:SparkContext, _tableName:String, collectionName:String): RDD[Tweet] = {
		val _cleanTweetColFam: String = "clean-tweet"
		val _tweetColFam : String =     "tweet"
		val _metadataColFam : String = "metadata"

		val _cleanTweetCol : String =   "clean-text-cla"
		val _tweetCol : String =        "text"
		val _peopleCol : String =       "sner-people"
		val _locationsCol : String =    "sner-location"
		val _orgCol : String =          "sner-organizations"  
		val _hashtagsCol : String =     "hashtags"
		val _longurlCol : String =      "long-url"
		val _collectionNameCol : String = 	"collection-name"
		val _docTypeCol : String = 			"doc-type"

		val connection = ConnectionFactory.createConnection()
		val table = connection.getTable(TableName.valueOf(_tableName))
		val scanner = new Scan()
		scanner.setMaxResultSize(1000)
		scanner.addColumn(Bytes.toBytes(_tweetColFam),Bytes.toBytes(_tweetCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_peopleCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_locationsCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_orgCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_hashtagsCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_longurlCol))
		scanner.addColumn(Bytes.toBytes(_metadataColFam), Bytes.toBytes(_docTypeCol))
		scanner.addColumn(Bytes.toBytes(_metadataColFam), Bytes.toBytes(_collectionNameCol))

		//filter for tweets and same collection name
		val filterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);

		println("Filter: Keeping Collection Name == " + collectionName)
		val filterCollect = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_collectionNameCol), CompareOp.EQUAL , Bytes.toBytes(collectionName));
		filterCollect.setFilterIfMissing(true);	//filter all rows that do not have a collection name
		filterList.addFilter(filterCollect);

		println("Filter: Keeping Doc Type == tweet")
		val filterTweet = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_docTypeCol), CompareOp.EQUAL , Bytes.toBytes("tweet"));
		filterTweet.setFilterIfMissing(true);	//filter all rows that are not marked as tweets
		filterList.addFilter(filterTweet);

        val filterEmptyCleanText = new SingleColumnValueFilter(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetCol), CompareOp.NOT_EQUAL , Bytes.toBytes(""));
		filterEmptyCleanText.setFilterIfMissing(true);	//filter all rows without a clean text tweet
		filterList.addFilter(filterEmptyCleanText);

		scanner.setFilter(filterList);

		sc.parallelize(table.getScanner(scanner).map(result => {
			val textcell = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetCol))
			val rawcell = result.getColumnLatestCell(Bytes.toBytes(_tweetColFam), Bytes.toBytes(_tweetCol))
			val peoplecell = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_peopleCol))
			val locationcell = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_locationsCol))
			val orgcell = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_orgCol))
			val hashtagcell = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_hashtagsCol))
			val longurlcell = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_longurlCol))


			val words = if(textcell != null) Bytes.toString(textcell.getValueArray, textcell.getValueOffset, textcell.getValueLength) else ""
			val people = if(peoplecell != null)	Bytes.toString(peoplecell.getValueArray, peoplecell.getValueOffset, peoplecell.getValueLength) else ""
			val locations = if(locationcell != null) Bytes.toString(locationcell.getValueArray, locationcell.getValueOffset, locationcell.getValueLength) else ""
			val orgs = if(orgcell != null) Bytes.toString(orgcell.getValueArray, orgcell.getValueOffset, orgcell.getValueLength) else ""
			val hashtags = if(hashtagcell != null) Bytes.toString(hashtagcell.getValueArray, hashtagcell.getValueOffset, hashtagcell.getValueLength) else ""
			val longurl = if(longurlcell != null) Bytes.toString(longurlcell.getValueArray, longurlcell.getValueOffset, longurlcell.getValueLength) else ""


			val combinewords = words + " " + people + " " + locations + " " + orgs + " " + hashtags + " " + longurl
			println ("Combinedwords: " + combinewords)

			val rawwords = if(rawcell != null) Bytes.toString(rawcell.getValueArray, rawcell.getValueOffset, rawcell.getValueLength) else ""
			var key = Bytes.toString(result.getRow())
			println("Label this tweetID: " + key + " | RAW: "+rawwords)
			val label = Console.readInt().toDouble
			Tweet(key, words, Option(label))
		}).toList)
	}
}

