package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{IDF, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    val data = spark.read.parquet("prepared_trainingset")

    val regexTokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(regexTokenizer.getOutputCol)
      .setOutputCol("filtered")


    val countVectorizer = new CountVectorizer()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("countVectorized")

    val idf = new IDF().setInputCol("countVectorized").setOutputCol("tfidf")

    val stringIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")


        /** 6eme stage: Indexer les currencies **/

    val stringIndexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign", "hours_prepa","goal","country_indexed","currency_indexed"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    val pipeline = new Pipeline()
      .setStages(Array(regexTokenizer, stopWordsRemover, countVectorizer, idf, stringIndexer, stringIndexer2, vectorAssembler, lr))


    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))


    val paramGrid = new ParamGridBuilder()
      .addGrid(countVectorizer.minDF,Array(55.0,75.0,95.0))
      .addGrid(lr.regParam, Array(0.0000001, 0.00001,0.001,0.1))
      .build()

    val mce = new MulticlassClassificationEvaluator()
      .setPredictionCol("predictions")
      .setLabelCol("final_status")
      .setMetricName("f1")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(mce)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)


    val model = trainValidationSplit.fit(trainingData)

    val df_WithPredictions = model.transform(testData)

    val f1=mce.evaluate(df_WithPredictions)
    print("f1 score is " + f1)

    df_WithPredictions.select("final_status","predictions").show()

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    model.write.overwrite().save("myModel")

  }
}
