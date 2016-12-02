package org.pfperez.matrices

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD

object DistributedMultiplicationMapReduceStyle extends App {

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("Matrix multiplication")
    .set("matrixAPath", "input_files/matrices/A.txt")
    .set("matrixBPath", "input_files/matrices/B.txt")

  val sc = new SparkContext(conf)

  /**
    * Builds a RDD of arrays of double values from a TXT file
    * where each line represents a row of a matrix of Doubles
    *
    * @param path Path to the input file
    * @return RDD[Array[Double]] containing the matrix rows.
    */
  def loadMatrix(path: String): RDD[Array[Double]] = sc.textFile(path).map(_.split(" ").map(_.toDouble))

  /**
    * Labels each Double value with its position in the 2D matrix
    * @param matrix 2D Double values matrix
    * @return 2D Double values, labeled with their position in the matrix.
    */
  def withIndex(matrix: RDD[Array[Double]]): RDD[Array[((Long, Long), Double)]] = {
    matrix.zipWithIndex map { row => // zip with index add each row index for every RDD element (row) -> i
      // The seam approach is used to label each Double value within each row -> j ...
      // i & j compose a position tuple which is then added to each value as a tuple of position-value.
      row._1.zipWithIndex map { case (v, j) => (row._2,j.toLong) -> v }  // v is transformed into ((i,j), v)
    }
  }

  /**
    * Flattens a matrix RDD so each entry is then a position (i,j) in the matrix
    * and the value at that position.
    *
    * @param matrix In the form of a RDD of Arrays of Doubles
    * @return The same matrix in the form of a Key-Value pairs RDD where the keys
    *         are the position of values in the matrix and the values are the elements at that position.
    */
  def flattenedWithIndex(matrix: RDD[Array[Double]]): RDD[((Long, Long), Double)] = {
    matrix.zipWithIndex flatMap { row => // Each array is iterated unsing the bind (flat map) monad ...
      // ... operator, that means that each element of:
      row._1.zipWithIndex map { case (v, j) => (row._2, j.toLong) -> v } //... iterable will become ...
      //... an entry in the resulting RDD.
      // This iterable produces a (position, value) pair for each cell in the row.
    }
  }

  /**
    *  Builds a list of list of doubles matrix representation from a map of positions to values.
    * @param withIndexMatrix flattened matrix RDD
    * @return tabular format matrix RDD
    */
  def unflattened(withIndexMatrix: Map[(Long, Long), Double]): List[List[Double]] =
    (0L to withIndexMatrix.keys.map(_._1).max toList) map { i =>
      (0L to withIndexMatrix.keys.map(_._2).max toList) map { j =>
        withIndexMatrix(i, j)
      }
    }

  /**
    * Perform a classical transpose matrix operation over a ...
    * @param fm matrix given in the (position, value) RDD format.
    * @return the transpose matrix in the same format.
    */
  def transposeFlattened(fm: RDD[((Long, Long), Double)]) = fm map { case ((i, j),v) => ((j,i), v)}

  def printMatrix(m: Seq[Seq[_]]) = m.foreach(row => println(row mkString " "))

  // The following code performs a distributed
  // matrix multiplication in the Spark cluster.

  // Load operands from files in the RDD of arrays of values format...
  // and transform them into (position, value) format.
  val A = flattenedWithIndex(loadMatrix(conf.get("matrixAPath")))
  val B = flattenedWithIndex(loadMatrix(conf.get("matrixBPath")))

  // The following operation generates a RDD with the left side elements of
  // each individual multiplication when (A*B)[i][j] elements get computed.
  // That is: j -> A[0][j], j -> A[1][j], ..., j -> A[n-1][j]
  val fromA = A.map { case ((i, j), v: Double) => j -> ((i,j),v) }
  // `fromB` follows the same pattern, now the key is the i position for a fixed j
  // That is: i -> A[i][0], i -> A[i][1], ..., i -> A[i][m-1]
  val fromB = B.map { case ((i, j), v: Double) => i -> ((i,j),v) }

  /**
    * The join by key of `fromA` & `fromB`, generates a RDD with all the pairs of
    * values whose pair-elements should be multiplied when computing A*B as:
    *
    * (A*B)[i][j] = A[i][0]*B[j][0] + A[i][1]*B[j][1] + ... + A[i][n-1]*B[j][m-1]
    *
    * So `cross` contains all pairs of operands from the summations above described:
    *
    * (A[i][0], B[j][0])
    * (A[i][1], B[j][1])
    * ...
    * (A[i][n-1], B[j][m-1])
    * (A[i][0], B[j+1][0])
    * (A[i][1], B[j+1][1])
    * ...
    * (A[i][n-1], B[j+1][m-1])
    * ...
    *
    * The result key is as `k` in the traditional O(n^3) algorithm computation
    *
    * for(i in [0,n-1] )
    *  for(j in [0,m-1])
    *   result[i][j] = 0
    *   for(k in [0, noCols(A)-1])
    *     result[i][j] += A[i][k]*B[k][j]
    *
    */
  val cross = fromA.join(fromB)

  /**
    * Then `cross` can be easily leveraged to build the solution as
    * all summations elements are provided by multiplying its tuples of values elements.
    *
    * That solution is, naturally, in the format of (position, value) Key-Value RDD
    */
  val res = cross map {
    case (k, (((ai,aj), va: Double), ((bi,bj), vb: Double))) => (ai, bj) -> va * vb
  } reduceByKey { case (a: Double, b: Double) => a + b }

  // The result is collected and transformed into tabular format.
  val resMatrix = unflattened(res.collect().toMap)

  printMatrix(resMatrix)

  sc.stop()

}
