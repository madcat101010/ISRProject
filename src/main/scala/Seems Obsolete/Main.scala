/*
Robin Yang
15 OCT 2017
Main.scala

Top level framework for the class cluster.
Flow of work:
1) load top level classifier models. 1 for website and 1 for tweets (create/train a model if not found)
2) check hbase table for new tweets/websites to classify
3) classify tweets first
 a) classify tweet with top level classifier model and store the class
 b) HEIR: Load subcategory models... maybe top 3? train if not exist but there is dataset for train/test
 c) HEIR: classify subcategory model and stoe the subclass
4)  classify websites next
 a) classify with top level classifer model and store the class
 b) HEIR: load appropriate subcategory model
 c) HEIR: classify subcategory model and store the subclass
5) wait for more data...

Top level model stored in the following format TOP_<alg_type>_<website/tweets>.model
All sub-class models are stored in the following format <topClassID>_<alg_type>_<website/tweets>.model
i.e. TOP_LR_website.model is the top level linear regression model for new website data. 5_LR_website.model is the model for top class id 5 of websites,
The class ID is just an number. the number can be replaced with the actual class i.e. 1 = shootings, 2 = hurricanes, etc.

Folders:
./data/ should be pushed to the cluster nodes as data for them. contains a lot of data...
./data/training/website/ website data for training website models
./data/training/tweet/ tweets for training tweet models
./data/testing/tweet/ tweets for testing the trained tweet model
./data/testing/website/ website data for testing the trained website models
./data/trainedModel/website/ trained website models
./data/trainedMOdel/tweet/ trained tweet models
*/
package isr.project

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Word2VecClassifier
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Main {

	

	//args = [  ]
	def main(args: Array[String]) {
		if (args.length > 0){
			System.err.println("Usage: Main  (no arguments allowed!)")
			System.exit(-1)
		}
		if (args.length == 0) {

			Logger.getLogger("org").setLevel(Level.OFF)
			Logger.getLogger("akka").setLevel(Level.OFF)
			

			val start = System.currentTimeMillis()
			val sc = new SparkContext()
			val readTweets = DataRetriever.retrieveTweets(args, sc)
			val end = System.currentTimeMillis()
			println(s"Took ${(end - start) / 1000.0} seconds for the whole process.")

		}
	}
}


