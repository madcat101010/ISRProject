#!/bin/sh

if [ "$#" -ne 1 ]
then
	  echo "Usage: buildAndRunLocal.sh <collection number>"
	    exit 1
fi


echo building the project
sbt package

echo running the script on collection "$1"
spark-submit --master local --driver-memory 20G --jars stanford-corenlp/jars/stanford-corenlp-3.4.1-models.jar,stanford-corenlp/jars/stanford-corenlp-3.4.1.jar,hbase-rdd_2.10/jars/hbase-rdd_2.10-0.7.1.jar --class isr.project.SparkGrep target/scala-2.10/sparkgrep_2.10-1.0.jar "$1" 9
#spark-submit  --jars stanford-corenlp/jars/stanford-corenlp-3.4.1-models.jar,stanford-corenlp/jars/stanford-corenlp-3.4.1.jar --class isr.project.SparkGrep target/scala-2.10/sparkgrep_2.10-1.0.jar "local[*]" "$1" "$2" 9
