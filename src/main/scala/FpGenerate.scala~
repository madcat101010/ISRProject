/**
  * Created by saur6410 on 10/1/16.
  */

package isr.project

import org.apache.spark.SparkContext
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

object FpGenerate {

  def generateFrequentPatterns(inputFilename: String, sc: SparkContext): RDD[Array[String]] = {

    //val conf = new SparkConf().setAppName("SparkGrep").setMaster(args(0))
    //val sc = new SparkContext(conf)
    val data = sc.textFile(inputFilename)
    val transactions: RDD[Array[String]] = data.map(line => line.trim.split(' ').toList.distinct.toArray)

    return transactions

    val fpg = new FPGrowth()
      .setMinSupport(0.3)
      .setNumPartitions(10)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    return model.freqItemsets.map(x => x.items)
  }
}