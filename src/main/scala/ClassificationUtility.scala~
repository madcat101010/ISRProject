package org.apache.spark.mllib.linalg

import org.apache.spark.mllib.classification.LogisticRegressionModel

object ClassificationUtility {
  def predictPoint(dataMatrix: Vector, model: LogisticRegressionModel):
  (Double, Array[Double]) = {
    require(dataMatrix.size == model.numFeatures)
    val dataWithBiasSize: Int = model.weights.size / (model.numClasses - 1)
    val weightsArray: Array[Double] = model.weights match {
      case dv: DenseVector => dv.values
      case _ =>
        throw new IllegalArgumentException(
          s"weights only supports dense vector but got type ${model.weights.getClass}.")
    }
    var bestClass = 0
    var maxMargin = 0.0
    val withBias = dataMatrix.size + 1 == dataWithBiasSize
    val classProbabilities: Array[Double] = new Array[Double](model.numClasses)
    //classProbabilities{5} = -997778768.0
    (0 until model.numClasses - 1).foreach { i =>
      var margin = 0.0
      dataMatrix.foreachActive { (index, value) =>
        if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
      }
      // Intercept is required to be added into margin.
      if (withBias) {
        margin += weightsArray((i * dataWithBiasSize) + dataMatrix.size)
      }
      if (margin > maxMargin) {
        maxMargin = margin
        bestClass = i + 1
      }
      classProbabilities(i + 1) = 1.0 / (1.0 + Math.exp(-(margin - maxMargin)))
    }

    val sumProbabilities = classProbabilities.sum
    //Normalize probabilities
    (0 until model.numClasses - 1).foreach { i =>

      classProbabilities(i) = classProbabilities(i) / sumProbabilities
    }


    return (bestClass.toDouble, classProbabilities)
  }
}
