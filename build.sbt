name := "sparkGrep"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "1.6.0","org.apache.spark" %% "spark-mllib" % "1.6.0")

resolvers ++= Seq(
  "Cloudera repos" at "https://repository.cloudera.com/artifactory/cloudera-repos",
  "Cloudera releases" at "https://repository.cloudera.com/artifactory/libs-release"
)

libraryDependencies += "eu.unicredit" %% "hbase-rdd" % "0.7.1"

libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.4.1" artifacts (Artifact("stanford-corenlp", "models"), Artifact("stanford-corenlp"))

libraryDependencies ++= Seq(
  "org.apache.hbase" % "hbase" % "1.2.3",
  "org.apache.hbase" % "hbase-common" % "1.0.0-cdh5.5.1" % "provided",
  "org.apache.hbase" % "hbase-client" % "1.0.0-cdh5.5.1" % "provided",
  "org.apache.hbase" % "hbase-server" % "1.0.0-cdh5.5.1" % "provided",
  "org.json4s" %% "json4s-jackson" % "3.2.11" % "provided"
)
