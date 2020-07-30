package app

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Binarizer, VectorAssembler}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{DoubleType, StructType}


object RiskPredictor {


//  Data Loading
  case class Credit(
//                   Expected Schema
                     creditability: Double,
                     balance: Double, duration: Double, history: Double, purpose: Double, amount: Double,
                     savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double,
                     residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double,
                     credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double
                   )


  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0),
      line(1) , line(2), line(3), line(4), line(5),
      line(6) , line(7) , line(8), line(9) , line(10) ,
      line(11) , line(12) , line(13), line(14), line(15) ,
      line(16) , line(17) , line(18) , line(19) , line(20)
    )
  }

  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).map(_.map(_.toDouble))
  }

  def main(args: Array[String]): Unit = {

//    Data Loading Defined Schema for the CSV
    val schema = new StructType()
      .add("creditability",DoubleType,true)
      .add("balance",DoubleType,true)
      .add("duration",DoubleType,true)
      .add("history",DoubleType,true)
      .add("purpose",DoubleType,true)
      .add("amount",DoubleType,true)
      .add("savings",DoubleType,true)
      .add("employment",DoubleType,true)
      .add("instPercent",DoubleType,true)
      .add("sexMarried",DoubleType,true)
      .add("guarantors",DoubleType,true)
      .add("residenceDuration",DoubleType,true)
      .add("assets",DoubleType,true)
      .add("age",DoubleType,true)
      .add("concCredit",DoubleType,true)
      .add("apartment",DoubleType,true)
      .add("credits",DoubleType,true)
      .add("occupation",DoubleType,true)
      .add("dependents",DoubleType,true)
      .add("hasPhone",DoubleType,true)
      .add("foreign",DoubleType,true)


    val sc = new SparkContext("local[*]","Loan Credit Risk Prediction")
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._


//    val df = parseRDD(sc.textFile("D:\\Zone24x7\\Loan Credit Risk Prediction\\src\\main\\scala\\input\\germancredit.csv")).map(parseCredit).toDF().cache()
//    df.printSchema

    val df = sqlContext.read.format("csv")
      .option("header", "true")
      .schema(schema)
      .load("D:\\Zone24x7\\Loan Credit Risk Prediction\\src\\main\\scala\\input\\germancredit.csv")

//    Extract Information From the Data Frame
    df.describe().show()

    df.limit(5).show()

    df.describe("creditability").show()

    df.describe("balance").show()

    df.groupBy("creditability").avg("balance").show

    df.registerTempTable("TableCredits")
    sqlContext.sql("SELECT creditability, avg(balance) as AverageBalance, avg(amount) as AverageAmount, avg(duration) as AverageDuration, avg(credits) as AverageCredits  FROM TableCredits GROUP BY creditability ").show

//  Make the Prediction Model

//    Define Feature Columns
    val featureColumns = Array("balance",
    "duration",
    "history",
    "purpose",
    "amount",
    "savings",
    "employment",
    "instPercent",
    "sexMarried",
    "guarantors",
    "residenceDuration",
    "assets",
    "age",
    "concCredit",
    "apartment",
    "credits",
    "occupation",
    "dependents",
    "hasPhone",
    "foreign")

//    Compress Feature Columns into single column
     val vectorAssembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features")

    val assembledDataFrame = vectorAssembler.transform(df)

    assembledDataFrame.printSchema()

    assembledDataFrame.limit(10).show()

//    Create Label Column (Can use StringIndexer or Binarizer)
    var binarizedDF= new Binarizer().setThreshold(0.999).setInputCol("creditability").setOutputCol("label").transform(assembledDataFrame)

    binarizedDF.printSchema()

    binarizedDF.limit(10).show()


//    Split Data for Training and Testing
    var splits=binarizedDF.randomSplit(Array(0.8,0.2),seed =12345)
    var trainingSet=splits(0)
    var testingSet=splits(1)

//Training the Model
    val randomForestClassifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(5)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setSeed(12345)

    var pipeline=new Pipeline().setStages(Array(randomForestClassifier))
    var model=pipeline.fit(trainingSet)

    //  Get predictions
    var predictions=model.transform(testingSet)

    //  Display sample predictions
   var predictedDf = predictions.select("features","creditability","label","prediction")

    predictedDf.limit(10).show()

//    Write Predicted data to a JSON File
//    predictedDf.write
//      .json("predictions.json")

    //  Evaluate the model
    var evaluator=new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

    var accuracy=evaluator.evaluate(predictions)
    print(s"Accuracy is : $accuracy")

    model.write.overwrite().save("Risk_Prediction.model")

  }
}
