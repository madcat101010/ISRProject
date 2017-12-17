#!/bin/sh

#if [ "$#" -ne 2 ]
#then
#	  echo "Usage: buildAndRun.sh <collection number> <block size>"
#	    exit 1
#fi


echo building the project
sbt package

echo running the script on tweet solar eclipsed collection with block size hardcoded
spark-submit --master local --driver-memory 5G --jars jarlib/stanford-english-corenlp-3.8.0-models.jar,jarlib/stanford-corenlp-3.8.0.jar,jarlib/hbase-rdd_2.11-0.8.0.jar --class isr.project.SparkGrep target/scala-2.10/sparkgrep_2.10-1.0.jar "train" "tweet" "getar-cs5604f17" "eclipsedatasample1_cla" "2017EclipseSolar2017" "solareclipse" "2017EclipseSolar2017" "NOT2017EclipseSolar2017"
#"train" "tweet" "getar-cs5604f17" "getar-cs5604f17" "solareclipse" "solareclipse" "SolarEclipse" "NotSolarEclipse"
