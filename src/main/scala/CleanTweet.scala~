package isr.project
import java.io.PrintWriter
import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.util.CoreMap
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer
import scala.tools.nsc.interpreter.session.JIterator

//import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
/**
  * Created by Eric on 10/14/2016.
  * Will store the lemmatization a stopword removal code from the raw data
  */
object CleanTweet {

  def clean(args: Array[String], outName:String): Unit = {
    val inputFilename = args(1)
    val conf = new SparkConf().setAppName("SparkGrep").setMaster(args(0))
    val sc = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.ERROR)
    //Get the training data file passed as an argument
    val trainingFileInput = sc.textFile(inputFilename).map(x => x.split(";",2)(1))
    val keys = sc.textFile(inputFilename).map(x => x.split(";",2)(0))
    val cleaned = getCleanedTweets(trainingFileInput, sc)
    //val cleaned_sent = cleaned.map(arr => arr.mkString(" "))
    cleaned.zip(trainingFileInput).foreach(println)
    keys.zip(cleaned.zip(trainingFileInput)).foreach(println)
    new PrintWriter(outName) { write(keys.zip(cleaned).map(va => va._1.toString +  ";" +va._2.toString).collect()
      .mkString("\n")); close() }
  }
  def clean(tweets: RDD[Tweet],sc : SparkContext): RDD[Tweet] = {

    //val conf = new SparkConf()/*.setAppName("SparkGrep").setMaster("local[*]")*/
    //val sc = new SparkContext(/*conf*/)
    //tweets.foreach(println)
    val tweetsRDD = tweets
    val keys = tweetsRDD.map(tweet => (tweet.id,tweet.label))
    val values = tweetsRDD.map(tweet => tweet.tweetText)
    keys.zip(getCleanedTweets(values,sc)).map(va => Tweet(va._1._1.toString,va._2.toString,va._1._2))
  }
  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP)
  : Seq[String] = {
    val t = text.replaceAll("[#]", "")
    val doc = new Annotation(t)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    var s = new ListBuffer[CoreMap]()
    def addIt(x: CoreMap) : Unit = {
      s += x
    }
    def scalaIterator[T](it: JIterator[T]) = new Iterator[T] {
      override def hasNext = it.hasNext
      override def next() = it.next()
    }
    scalaIterator(doc.get(classOf[SentencesAnnotation]).iterator()).foreach(addIt)
    for (sentence <- s){
      val t = new ListBuffer[CoreLabel]() ;
      for (tw <- scalaIterator(sentence.get(classOf[TokensAnnotation]).iterator())){
        t += tw
      }
      for (token <- t){
        val lemma = token.get(classOf[LemmaAnnotation])
        if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
          lemmas += lemma.toLowerCase
        }
      }
    }
    lemmas
  }

  def isOnlyLetters(str: String): Boolean = {
    // While loop for high performance
    var i = 0
    while (i < str.length) {
      if (!Character.isLetter(str.charAt(i))) {
        return false
      }
      i += 1
    }
    true
  }
  def getCleanedTweets(tweets: RDD[String], sc: SparkContext) : RDD[String] = {
    val stopWords = sc.broadcast(
                              scala.io.Source.fromFile("data/stopwords.txt").getLines().toSet).value
    tweets.mapPartitions(it => {
      val pipeline = createNLPPipeline()
      it.map(t => plainTextToLemmas(t,stopWords,pipeline).mkString(" "))
    })
  }
  /*
  def cleanWord(word: String): String = {
    val pipeline = createNLPPipeline()

    def plainTweetToLemma(text: String, pipeline: StanfordCoreNLP)
    : String = {
      val doc = new Annotation(text)
      pipeline.annotate(doc)
      var lemma = "INVALID"
      val sentences = doc.get(classOf[SentencesAnnotation])
      for (sentence <- sentences.asScala;
           token <- sentence.get(classOf[TokensAnnotation]).asScala) {
        val lemma_t = token.get(classOf[LemmaAnnotation])
        if (lemma_t.length > 2 && isOnlyLetters(lemma_t)) {
          lemma = lemma_t
        }
      }
      lemma
    }
    plainTweetToLemma(word,pipeline)
  }*/
}

