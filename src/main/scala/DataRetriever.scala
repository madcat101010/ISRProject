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
import org.apache.spark.mllib.feature.{HashingTF, IDFModel, Word2Vec, Word2VecModel}

import scala.collection.mutable.ListBuffer
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
	val _cleanWebpageTextCol : String = "clean-text-profanity"

  def retrieveTweets(eventName:String, collectionName:String, _cachedRecordCount:Int, tableNameSrc:String, tableNameDest:String, sc: SparkContext): RDD[Tweet] = {
    //implicit val config = HBaseConfig()
		var _lrModelFilename = "./data/" + eventName +"_webpage_lr.model";
		var _word2VecModelFilename = "./data/" + tableNameSrc +"_table_w2v.model";
		

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
		val cell1 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetTextCol))
		val cell2 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetHashtags))
		val cell3 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerOrg))
		val cell4 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetLongURL))
		val cell5 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerPeople))
		val cell6 = result.getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerLoc))
			
		val key : String = Bytes.toString(result.getRow())
		var words : String = ""
		if(cell1 != null)
			words += (Bytes.toString(cell1.getValueArray, cell1.getValueOffset, cell1.getValueLength) + " ")
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
		var _lrModelFilename = "./data/" + eventName +"_webpage_lr.model";
		var _word2VecModelFilename = "./data/" + tableNameSrc +"_table_w2v.model";


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

		if( !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_metaDataColFam)) ){
			System.err.println("ERROR: Source webpage table missing required metadata column family!");
			return null;
		}

		if( !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_cleanWebpageColFam)) ){
			System.err.println("ERROR: Source webpage table missing required clean-webpage column family!");
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
          println("*********** Predicting the webpages now. *****************")
					val predictedTweets = Word2VecClassifier.predictClass(rddT, sc, word2vecModel, logisticRegressionModel)
          println("*********** Persisting the webpages now. *****************")

          val repartitionedPredictions = predictedTweets.repartition(12)
          DataWriter.writeTweets(repartitionedPredictions, tableNameDest)	//only writes to classificaiton column family, and Tweet data struct has row key

          predictedTweets.cache()
          val batchTweetCount = predictedTweets.count()
          println(s"The amount of webpages to be written is $batchTweetCount")
          val end = System.currentTimeMillis()
          totalRecordCount += batchTweetCount
          println(s"Took ${(end-start)/1000.0} seconds for This Batch.")
          println(s"This batch had $batchTweetCount webpages. We have processed $totalRecordCount webpages overall")
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


	def getTweetToTestAcc(sc:SparkContext, _tableName:String, collectionName:String, tweetLimit:Int): RDD[Tweet] = {
		val _cleanTweetColFam: String = "clean-tweet"
		val _tweetColFam : String =     "tweet"
		val _metadataColFam : String = "metadata"
        val _claColFam : String = "classification"

		val _cleanTweetCol : String =   "clean-text-cla"
		val _tweetCol : String =        "text"
		val _peopleCol : String =       "sner-people"
		val _locationsCol : String =    "sner-location"
		val _orgCol : String =          "sner-organizations"  
		val _hashtagsCol : String =     "hashtags"
		val _longurlCol : String =      "long-url"
		val _collectionNameCol : String = 	"collection-name"
		val _docTypeCol : String = 			"doc-type"
        val _rtCol : String =           "rt"
        val _claListCol : String = "classification-list"
        val _claProCol : String = "probability-list"

		val connection = ConnectionFactory.createConnection()
		val table = connection.getTable(TableName.valueOf(_tableName))
		val scanner = new Scan()
		scanner.addColumn(Bytes.toBytes(_tweetColFam),Bytes.toBytes(_tweetCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_peopleCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_locationsCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_orgCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_hashtagsCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_longurlCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_rtCol))
		scanner.addColumn(Bytes.toBytes(_metadataColFam), Bytes.toBytes(_docTypeCol))
		scanner.addColumn(Bytes.toBytes(_metadataColFam), Bytes.toBytes(_collectionNameCol))
		scanner.addColumn(Bytes.toBytes(_claColFam), Bytes.toBytes(_claListCol))
		scanner.addColumn(Bytes.toBytes(_claColFam), Bytes.toBytes(_claProCol))

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

        val filterRT = new SingleColumnValueFilter(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_rtCol), CompareOp.EQUAL , Bytes.toBytes("false"));
		filterRT.setFilterIfMissing(true);	//filter all rows with RT flag
		filterList.addFilter(filterRT);


		scanner.setFilter(filterList);

        var TP_tweet:Int = 0
        var FP_tweet:Int = 0
        var TN_tweet:Int = 0
        var FN_tweet:Int = 0

        var tweet_count:Int = 0
        var resultScanner = table.getScanner(scanner)
        var listTweet = new ListBuffer[Tweet]()
        var continueLoop = true
        var totalRecordCount: Long = 0
        while (continueLoop) {
          try {
            val result = resultScanner.next(100)
            if (result == null || result.isEmpty){
                println("No more results from scan. Finishing...")
              continueLoop = false
                    }
            else {
              val textcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetCol))
              val rawcell = result(0).getColumnLatestCell(Bytes.toBytes(_tweetColFam), Bytes.toBytes(_tweetCol))
              val peoplecell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_peopleCol))
              val locationcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_locationsCol))
              val orgcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_orgCol))
              val hashtagcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_hashtagsCol))
              val longurlcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_longurlCol))
              val clalistcell = result(0).getColumnLatestCell(Bytes.toBytes(_claColFam), Bytes.toBytes(_claListCol))
              val claprocell = result(0).getColumnLatestCell(Bytes.toBytes(_claColFam), Bytes.toBytes(_claProCol))
              


              val words = if(textcell != null) Bytes.toString(textcell.getValueArray, textcell.getValueOffset, textcell.getValueLength) else ""
              val people = if(peoplecell != null)	Bytes.toString(peoplecell.getValueArray, peoplecell.getValueOffset, peoplecell.getValueLength) else ""
              val locations = if(locationcell != null) Bytes.toString(locationcell.getValueArray, locationcell.getValueOffset, locationcell.getValueLength) else ""
              val orgs = if(orgcell != null) Bytes.toString(orgcell.getValueArray, orgcell.getValueOffset, orgcell.getValueLength) else ""
              val hashtags = if(hashtagcell != null) Bytes.toString(hashtagcell.getValueArray, hashtagcell.getValueOffset, hashtagcell.getValueLength) else ""
              val raw_longurl = if(longurlcell != null) Bytes.toString(longurlcell.getValueArray, longurlcell.getValueOffset, longurlcell.getValueLength) else ""
              val claList = if(clalistcell != null) Bytes.toString(clalistcell.getValueArray, clalistcell.getValueOffset, clalistcell.getValueLength) else ""
              val claPro = if(claprocell != null) Bytes.toString(claprocell.getValueArray, claprocell.getValueOffset, claprocell.getValueLength) else ""



              val longurl = raw_longurl.replaceAll("""<(?!\/?a(?=>|\s.*>))\/?.*?>""","")
              val combinewords = "[" + claList + "] [" + claPro + "]"


              println ("NUMBER OF LABELED TWEETS: " + tweet_count + " / "+tweetLimit)
              println ("Key: 1 => True Positive; 2 => False Positive; 3 => True Negative; 4 => False Negative")
              

              val rawwords = if(rawcell != null) Bytes.toString(rawcell.getValueArray, rawcell.getValueOffset, rawcell.getValueLength) else ""
              var key = Bytes.toString(result(0).getRow())
              println("For This tweet: "+rawwords)
              println("We label it as: "+combinewords)
              val label = Console.readInt().toDouble
              if (label == 1.0){
                TP_tweet += 1
              }
              else if (label == 2.0){
                FP_tweet += 1
              }
              else if (label == 3.0){
                TN_tweet += 1
              }
              else {
                FN_tweet += 1
              }

              println("")
              tweet_count = tweet_count + 1;
              listTweet += Tweet(key, rawwords + " | " + combinewords, Option(label))
              if (tweet_count >= tweetLimit){
                continueLoop = false
              }
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
        println("================TEST RESULTS=====================")
        println("TP Classified Tweets: "+TP_tweet+ " out of " + tweet_count)
        println("FP Classified Tweets: "+FP_tweet+ " out of " + tweet_count)
        println("TN Classified Tweets: "+TN_tweet+ " out of " + tweet_count)
        println("FN Classified Tweets: "+FN_tweet+ " out of " + tweet_count)
		sc.parallelize(listTweet.toList)
    }

/////////////////////////////////////////
	//////////////////////////////////


	def getTrainingTweets(sc:SparkContext, _tableName:String, collectionName:String, tweetLimit:Int): RDD[Tweet] = {
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
        val _rtCol : String =           "rt"

		val connection = ConnectionFactory.createConnection()
		val table = connection.getTable(TableName.valueOf(_tableName))
		val scanner = new Scan()
		scanner.addColumn(Bytes.toBytes(_tweetColFam),Bytes.toBytes(_tweetCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_peopleCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_locationsCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_orgCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_hashtagsCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_longurlCol))
		scanner.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_rtCol))
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

        val filterRT = new SingleColumnValueFilter(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_rtCol), CompareOp.EQUAL , Bytes.toBytes("false"));
		filterRT.setFilterIfMissing(true);	//filter all rows with RT flag
		filterList.addFilter(filterRT);


		scanner.setFilter(filterList);

        var tweet_count:Int = 0
        var resultScanner = table.getScanner(scanner)
        var listTweet = new ListBuffer[Tweet]()
        var continueLoop = true
        var totalRecordCount: Long = 0
        while (continueLoop) {
          try {
            val result = resultScanner.next(1)
            if (result == null || result.isEmpty){
                println("No more results from scan. Finishing...")
              continueLoop = false
                    }
            else {
              val textcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetCol))
              val rawcell = result(0).getColumnLatestCell(Bytes.toBytes(_tweetColFam), Bytes.toBytes(_tweetCol))
              val peoplecell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_peopleCol))
              val locationcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_locationsCol))
              val orgcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_orgCol))
              val hashtagcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_hashtagsCol))
              val longurlcell = result(0).getColumnLatestCell(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_longurlCol))


              val words = if(textcell != null) Bytes.toString(textcell.getValueArray, textcell.getValueOffset, textcell.getValueLength) else ""
              val people = if(peoplecell != null)	Bytes.toString(peoplecell.getValueArray, peoplecell.getValueOffset, peoplecell.getValueLength) else ""
              val locations = if(locationcell != null) Bytes.toString(locationcell.getValueArray, locationcell.getValueOffset, locationcell.getValueLength) else ""
              val orgs = if(orgcell != null) Bytes.toString(orgcell.getValueArray, orgcell.getValueOffset, orgcell.getValueLength) else ""
              val hashtags = if(hashtagcell != null) Bytes.toString(hashtagcell.getValueArray, hashtagcell.getValueOffset, hashtagcell.getValueLength) else ""
              val raw_longurl = if(longurlcell != null) Bytes.toString(longurlcell.getValueArray, longurlcell.getValueOffset, longurlcell.getValueLength) else ""


              val longurl = raw_longurl.replaceAll("""<(?!\/?a(?=>|\s.*>))\/?.*?>""","")
              val raw_combinewords = words + " " + people + " " + locations + " " + orgs + " " + hashtags + " " + longurl

              //remove tweet syntax
              val combinewords = raw_combinewords.replaceAll("[@#]","").replaceAll("[,]"," ")

              println ("NUMBER OF LABELED TWEETS: " + tweet_count + " / "+tweetLimit)
              println ("Combinedwords: " + combinewords)
              

              val rawwords = if(rawcell != null) Bytes.toString(rawcell.getValueArray, rawcell.getValueOffset, rawcell.getValueLength) else ""
              var key = Bytes.toString(result(0).getRow())
              println("Label this tweetID: " + key + " | RAW: "+rawwords)
              val label = Console.readInt().toDouble
              println("")
              tweet_count = tweet_count + 1;
              listTweet += Tweet(key, combinewords, Option(label))
              if (tweet_count >= tweetLimit){
                continueLoop = false
              }
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
		sc.parallelize(listTweet.toList)
    }

/////////////////////////////////////////

def trainW2VOnTable(tableNameSrc:String, sc: SparkContext): Word2VecModel = {

    val hbaseConf = HBaseConfiguration.create()
    val srcTable = new HTable(hbaseConf, tableNameSrc)
		
		//error check the table for columns
		if( !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_cleanTweetColFam)) ){
			System.err.println("ERROR: Source table missing required clean-tweet column family!");
			return null;
		}


		if( !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_metaDataColFam)) ){
			System.err.println("ERROR: Source table missing required metadata column family!");
			return null;
		}

		if( !srcTable.getTableDescriptor().hasFamily(Bytes.toBytes(_cleanWebpageColFam)) ){
			System.err.println("ERROR: Source table missing required clean-webpage column family!");
			return null;

		}

    // scan over only the tweets
    val Tscan = new Scan()
	  
		// MUST scan the column to filter using it... else it assumes column does not exist and will auto filter if setFilterIfMissing(true) is set.
		Tscan.addColumn(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol));
		Tscan.addColumn(Bytes.toBytes(_tweetColFam), Bytes.toBytes(_tweetUniqueIdCol));
		Tscan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetTextCol));
		Tscan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetHashtags));
		Tscan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerOrg));
		Tscan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetLongURL));
		Tscan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerPeople));
		Tscan.addColumn(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetSnerLoc));

		//filter for only same collection, is tweet, has clean text, and not classified ... uncomment when table has the missing fields
		val TfilterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);

		println("Filter: Keeping Doc Type == tweet")
		val filterTweet = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol), CompareOp.EQUAL , Bytes.toBytes("tweet"));
		filterTweet.setFilterIfMissing(true);	//filter all rows that are not marked as tweets
		TfilterList.addFilter(filterTweet);

		println("Filter: Keeping Clean Text != ''")
		val filterNoClean = new SingleColumnValueFilter(Bytes.toBytes(_cleanTweetColFam), Bytes.toBytes(_cleanTweetTextCol), CompareOp.NOT_EQUAL , Bytes.toBytes(""));	//note compareOp vs compareOperator depending on hadoop version
		filterNoClean.setFilterIfMissing(true);	//filter all rows that do not have clean text column
		TfilterList.addFilter(filterNoClean);

		Tscan.setFilter(TfilterList);

    // add caching to increase speed
		Tscan.setCaching(200000)
    val TresultScanner = srcTable.getScanner(Tscan)

		println(s"Caching Info:${Tscan.getCaching} Batch Info: ${Tscan.getBatch}")
		println("Scanning tweet results now.")

		var continueLoop = true
		var totalRecordCount: Long = 0

		var AllTRDD : org.apache.spark.rdd.RDD[Tweet] = sc.emptyRDD;
    while (continueLoop) {
      try {
        println("Getting next batch of results now.")
        val start = System.currentTimeMillis()
        val results = TresultScanner.next(200000)
        if (results == null || results.isEmpty){
        	println("No more results from scan. Doing websites...")
          continueLoop = false
				}
        else {
          val resultTweets = results.map(r => rowToTweetConverter(r))

          val rddT = sc.parallelize(resultTweets)
					AllTRDD = AllTRDD.union(rddT)

          val batchTweetCount = rddT.count()
          println(s"The amount of tweets loaded is $batchTweetCount")
          val end = System.currentTimeMillis()
          totalRecordCount += batchTweetCount
          println(s"Took ${(end-start)/1000.0} seconds for This Batch.")
          println(s"We have loaded $totalRecordCount tweets overall")
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


    println(s"Total tweet record count:$totalRecordCount")
    TresultScanner.close()
    //val interactor = new HBaseInteraction(_tableName)


		// scan over only the collection
		val Wscan = new Scan()
	  // MUST scan the column to filter using it... else it assumes column does not exist and will auto filter if setFilterIfMissing(true) is set.
		Wscan.addColumn(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataCollectionNameCol));
		Wscan.addColumn(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol));
		Wscan.addColumn(Bytes.toBytes(_cleanWebpageColFam), Bytes.toBytes(_cleanWebpageTextCol));

		//filter for only same collection, is tweet, has clean text, and not classified ... uncomment when table has the missing fields
		val WfilterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);


		println("Filter: Keeping Doc Type == webpage")
		val filterWeb = new SingleColumnValueFilter(Bytes.toBytes(_metaDataColFam), Bytes.toBytes(_metaDataTypeCol), CompareOp.EQUAL , Bytes.toBytes("webpage"));
		filterWeb.setFilterIfMissing(true);	//filter all rows that are not marked as tweets
		WfilterList.addFilter(filterWeb);

		println("Filter: Keeping Clean Text != ''")
		val filterNoCleanW = new SingleColumnValueFilter(Bytes.toBytes(_cleanWebpageColFam), Bytes.toBytes(_cleanWebpageTextCol), CompareOp.NOT_EQUAL , Bytes.toBytes(""));	//note compareOp vs compareOperator depending on hadoop version
		filterNoCleanW.setFilterIfMissing(true);	//filter all rows that do not have clean text column
		WfilterList.addFilter(filterNoCleanW);
		
		Wscan.setFilter(WfilterList);


		// add caching to increase speed
		Wscan.setCaching(2000)
		val WresultScanner = srcTable.getScanner(Wscan)

		println(s"Caching Info:${Wscan.getCaching} Batch Info: ${Wscan.getBatch}")
		println("Scanning results now.")
		
		var AllWRDD : org.apache.spark.rdd.RDD[Tweet] = sc.emptyRDD;
		
		continueLoop = true
		totalRecordCount = 0
		while (continueLoop) {
			try {
				println("Getting next batch of results now.")
				val start = System.currentTimeMillis()

        val results = WresultScanner.next(2000)

        if (results == null || results.isEmpty){
					println("No more results from scan. Finishing...")
					continueLoop = false
				}
        else {
          println(s"Result Length:${results.length}")
          val resultTweets = results.map(r => rowToWebpageConverter(r))
          val rddT = sc.parallelize(resultTweets)
					AllWRDD = AllWRDD.union(rddT)
          val batchTweetCount = rddT.count()
          println(s"The amount of webpages to loaded is $batchTweetCount")
          val end = System.currentTimeMillis()
          totalRecordCount += batchTweetCount
          println(s"Took ${(end-start)/1000.0} seconds for This Batch.")
          println(s"We have loaded $totalRecordCount webpages overall")
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

    println(s"Total webpage record count:$totalRecordCount")
    WresultScanner.close()
		
		//combine and train word2vec model!
		AllTRDD = AllTRDD.union(AllWRDD)
    
		def cleanHtml(str: String) = str.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", "");
    def cleanTweetHtml(sample: Tweet) = sample copy (tweetText = cleanHtml(sample.tweetText));
    def cleanWord(str: String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W", "")).reduceOption((x, y) => s"$x $y");
    def wordOnlySample(sample: Tweet) = sample copy (tweetText = cleanWord(sample.tweetText).getOrElse(""));

    val cleanTrainingTweets = AllTRDD map cleanTweetHtml;

    val wordOnlyTrainSample = cleanTrainingTweets map wordOnlySample

    // Word2Vec
    val samplePairs = wordOnlyTrainSample.map(s => s.id -> s)
    val reviewWordsPairs: RDD[(String, Iterable[String])] = samplePairs.mapValues(_.tweetText.split(" ").toIterable)

    val word2vecModel = new Word2Vec().fit(reviewWordsPairs.values)

    return word2vecModel

  }



}

