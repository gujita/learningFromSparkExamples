/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.log4j.{Level, Logger}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * A simple example demonstrating model selection using CrossValidator.
 * This example also demonstrates how Pipelines are Estimators.
 *
 * Run with
 * {{{
 * bin/run-example ml.ModelSelectionViaCrossValidationExample
 * }}}
 * 
 */

/*看一下run-example脚本：
if [ -z "${SPARK_HOME}" ]; then
  source "$(dirname "$0")"/find-spark-home
fi

export _SPARK_CMD_USAGE="Usage: ./bin/run-example [options] example-class [example args]"
exec "${SPARK_HOME}"/bin/spark-submit run-example "$@"
*/
object ModelSelectionViaCrossValidationExample {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.hadoop").setLevel(Level.WARN)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val spark = SparkSession
      .builder
      .master("local")
      .appName("ModelSelectionViaCrossValidationExample")
      .getOrCreate()

    // $example on$
    // Prepare training data from a list of (id, text, label) tuples.
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    // 配置ML pipeline, 该流水线包括三个阶段: tokenizer（分词器）, hashingTF, lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10) // 迭代次数
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr)) // 函数原型：setStages(value: Array[_ <: PipelineStage]): Pipeline.this.type

    // 使用ParamGridBuilder 来构建 a grid of parameters to search over.
    // hashingTF.numFeatures设置三个值， lr.regParam设置两个值
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    // 这样一来，对于CrossValidator，就可以评估6个参数，也可以认为可以学习6种模型（模型运行是使用“fit”方法，2.0.0版本加入）
    // hashingTF:目前使用Austin Appleby的MurmurHash 3算法————将词映射为词频
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000)) // Features的数量，默认为2的18次方，TODO：数组？
      .addGrid(lr.regParam, Array(0.1, 0.01)) // 正则化参数，默认0.0
      .build() // 

    // 现在将Pipeline视为一个Estimator（进行算法或管道调整）, 把它包裹进一个CrossValidator实例
    // 这会在所有的Pipeline stages中调整参数
    // 所有的模型选择器都需要：
    // 一个Estimator【之前设置的流水线】, 一个Estimator参数集【之前设置的paramGrid】, 一个Evaluator【评估者：衡量拟合模型对延伸测试数据有多好的度量】.
    // 这里的BinaryClassificationEvaluator是默认的(我修改为了)
    // 模型评估工具有————回归：RegressionEvaluator；二进制数据：BinaryClassificationEvaluator；多类问题：MulticlassClassificationEvaluator
    val cv = new CrossValidator() // 模型选择工具有： CrossValidator和TrainValidationSplit
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator) // 默认值是BinaryClassificationEvaluator
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)  // 生成2个（训练，测试）数据集对，其中训练数据占比2/3，测试数据占1/3
      //详：【参数值必须>=2，默认3】。数据集对是不重叠的随机拆分，每对中的测试数据仅测试一次。
      //所以数据集大时，我们可以设置得高一些，但是数据集小时，可能会过拟合。TODO：不能整除怎么办？

    // 运行交叉验证，选择出最好的参数集
    // fit函数详见：https://github.com/apache/spark/blob/v2.1.1/mllib/src/main/scala/org/apache/spark/ml/tuning/CrossValidator.scala
    // 对于上面的setNumFolds，fit函数会调用MLUtils的kFold方法进行拆分（org.apache.spark.mllib.util.MLUtils）
    val cvModel = cv.fit(training) // 对于这里，返回一个CrossValidatorModel
    /*源码106、107已经将我们的训练集和测试集缓存*/

    // 准备测试集
    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    val test2 = spark.createDataFrame(Seq(
      (4L, "spark wang j k"),
      (5L, "l m hai"),
      (6L, "mapreduce spark wang"),
      (7L, "apache hadoop hai")
    )).toDF("id", "text")

    // 使用刚才学习出来的模型对测试集进行预测
    cvModel.transform(test) // transform对数据进行转换，返回一个DataFrame。"probability"和"prediction"
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }

    println("===========我的test2===========")
    cvModel.transform(test2).show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
