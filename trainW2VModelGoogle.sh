#!/bin/sh

#if [ "$#" -ne 2 ]
#then
#	  echo "Usage: buildAndRun.sh <collection number> <block size>"
#	    exit 1
#fi


#echo building the project
#sbt package
#echo running the script on tweet solar eclipsed collection with block size hardcoded
#spark-submit --master local --driver-memory 20G --jars jarlib/stanford-english-corenlp-3.8.0-models.jar,jarlib/stanford-corenlp-3.8.0.jar,jarlib/hbase-rdd_2.11-0.8.0.jar --class isr.project.SparkGrep target/scala-2.10/sparkgrep_2.10-1.0.jar "train" "w2v" "eclipsedatasample1" "eclipsedatasample1_cla" "solareclipse" "solareclipse" "SolarEclipse" "NotSolarEclipse"


echo DEPRECATED:
echo See final report about training Google News pre-trained Word2Vec model conversion
echo This no longer works.