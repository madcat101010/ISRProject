/*


*/

package isr.project

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD


/**
 * Created by saur6410 on 10/2/16.
 */

class Greeter(message: RDD[Array[String]]) {
    def SayHi() = println(message)
}

class FeatureGenerator(tweets: RDD[Tweet]) {



  val _tweets = tweets
  val tweetLength:Int = 70
  var _wcpClassesDict:Array[collection.Map[String, Double]] = null
  var _totalVocabularySize = 5000

  def GenerateClassMaps(classLabel: Int, totalVocabularySize:Int): collection.Map[String, Double] = {

    //Get all the tweets for the class label
    val classTweets = _tweets.filter(_.label.get == classLabel.toDouble).map(_.tweetText)
    //Flap map all the tweets into a single list.
    val flatTweets = classTweets.flatMap(a => a.split(" "))

    // give each word a count of 1
    val wordT = flatTweets map (x => (x.toLowerCase,1))
    //sum up the counts for each word
    val classMapWithCounts = wordT reduceByKey((a, b) => a + b)

    //Generate the word probabilities for each term
    val classVocabularySize = flatTweets.count()

    //val t = classMapWithCounts.mapValues(v => (v -> v.toDouble/5))

    //val classTermProbabilities = wordMap map(word => (word._1 -> (1+word._2)/(totalVocabularySize + classVocabularySize)))
    val classTermProbabilities = classMapWithCounts.map(word => (word._1 -> (1 + word._2).toDouble/(totalVocabularySize + classVocabularySize)))



    //This method would return a map of a word, and its corresponding count
    //return null
    return classTermProbabilities.collectAsMap()
  }

  def InitializeFeatures(featureType: String, numberOfClasses: Int) = featureType match {
    case "wcp" => {

      //Initialize an array to accommodate a dictionary for all classes
      val classesDict = new Array[collection.Map[String, Double]](numberOfClasses)
      _totalVocabularySize = 5000
      for (i <- 1 to numberOfClasses) {
        classesDict(i - 1) = GenerateClassMaps(i-1, _totalVocabularySize)
      }
      _wcpClassesDict = classesDict


      val uniqueVocabularyWords = _wcpClassesDict.map(x => x.toIterable)
      //Calculate the gini-coefficients

    }



  }

  def getFeatures(featureType: String, documentBody: String) = featureType match {
    case "fpm" => {
      //Load the FPM List of Lists
      //Initialize a new array of booleans based on the FPM List length
      //Iterate through the FPM List
      //For each fpm record, find the intersection with the documentBody list(maybe?)
        //If the intersection set size is the same as the fpm record size, set the boolean to 1.0
        //Add the boolean to the array of booleans that is being composed.
      //Return the array of booleans as a Dense Vector

      //ToDo: Remove this stubbed implementation
      var issac, hurricane, weather = 0.0
      if(documentBody.contains("issac"))
        issac = 1.0
      if(documentBody.contains("hurricane"))
        hurricane = 1.0
      if(documentBody.contains("weather"))
        weather = 1.0

      Vectors.dense(issac, hurricane, weather)
    }
    case "wcp" => {

      Vectors.dense(1.0, 0.0, 3.0)
    }

  }


  def PadFeatureArray(features: Array[Double]): Array[Double] = {

    val paddedFeatures = new Array[Double](tweetLength)
    for (i <- 1 to features.length) {
        paddedFeatures(i - 1) = features(i -1)
    }
    return paddedFeatures
  }

}
