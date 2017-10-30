#!/bin/sh
export HADOOP_CONF_DIR=/etc/hadoop/conf
sbt package
spark-submit --master yarn --num-executors 8 --executor-cores 4 --driver-memory 4G --executor-memory 4G --class ExperimentRunner target/scala-2.10/sparkgrep_2.10-1.0.jar

