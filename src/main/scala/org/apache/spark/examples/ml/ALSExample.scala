// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
// $example off$
import org.apache.spark.sql.SparkSession

object ALSExample {

  // $example on$
/**  当你声明了一个 case class（样例类）时，scala会帮助我们做下面几件事情：
  1 构造器中的参数如果不被声明为var的话，它默认的话是val类型的，但一般不推荐将构造器中的参数声明为var
  2 自动创建伴生对象，同时在里面给我们实现子apply方法，使得我们在使用的时候可以不直接显示地new对象
  3 伴生对象中同样会帮我们实现unapply方法，从而可以将case class应用于模式匹配
  4 实现自己的toString、hashCode、copy、equals方法
  */
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  // 数据示例：0::2::3::1424380312
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
  //    assert() 或 assume() 方法在对中间结果或私有方法的参数进行检验，不成功则抛出 AssertionError 异常
    assert(fields.size == 4)
  //      返回分割好的Rating对象,注意用户Id列以及产品id列，必须是整数（如果是其他数值类型，只要在整形范围内，都会转为integers）
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  // $example off$

  def main(args: Array[String]) {
    Logger.getLogger("org.apache.hadoop").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder
      .master("local")
      .appName("ALSExample")
      .getOrCreate()
    import spark.implicits._

    // $example on$
    val ratings = spark.read.textFile("data/mllib/als/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()
    ratings.printSchema()
    ratings.show(5)
//    切分训练集和测试集
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setRank(8) //  对应ALS模型中的因子个数，即矩阵分解出的两个矩阵的新的行/列数，即A≈UVT,k<<m,n中的k。
      .setMaxIter(20) // 默认10
      .setRegParam(0.01) // 正则化参数
      .setImplicitPrefs(true) // 使用隐式反馈
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(training)

    // 通过计算测试集上的均方根误差来评估模型
    // transform操作会增加一个prediction列
    val predictions = model.transform(test)
    println("=================predictions")
    predictions.printSchema()
    predictions.show(7)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
    // $example off$
    spark.stop()
  }
}
// scalastyle:on println

