#!/bin/sh

#if [ "$#" -ne 2 ]
#then
#	  echo "Usage: buildAndRun.sh <collection number> <block size>"
#	    exit 1
#fi


echo building the project
sbt package

echo running the script on collection "$1" with block size "$2"
spark-submit --master local --driver-memory 5G --jars jarlib/stanford-english-corenlp-3.8.0-models.jar,jarlib/stanford-corenlp-3.8.0.jar,jarlib/hbase-rdd_2.11-0.8.0.jar --class isr.project.SparkGrep target/scala-2.10/sparkgrep_2.10-1.0.jar "train" "tweet" "dummy" "dummy" "solareclipse" "SolarEclipse" "NOTSolarEclipse"
#spark-submit  --jars stanford-corenlp/jars/stanford-corenlp-3.4.1-models.jar,stanford-corenlp/jars/stanford-corenlp-3.4.1.jar --class isr.project.SparkGrep target/scala-2.10/sparkgrep_2.10-1.0.jar "local[*]" "$1" "$2" 9

