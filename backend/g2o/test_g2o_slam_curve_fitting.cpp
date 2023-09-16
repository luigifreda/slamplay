#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h> 
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

//Vertices for Curve Models, Template Parameters: Optimizing Variable Dimensions and Data Types
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

  //reset
  virtual void setToOriginImpl() override {
    _estimate << 0, 0, 0; 
  }

  //renew
  virtual void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector3d(update);
  }

  //Save and read: leave blank
  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}
};

//ErrorModel template parameters: observation dimension, type, connected vertex type
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

  //Calculate curve model error
  virtual void computeError() override {
    const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
  }

  //Compute the Jacobian matrix
  virtual void linearizeOplus() override {
    const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    _jacobianOplusXi[0] = -_x * _x * y;
    _jacobianOplusXi[1] = -_x * y;
    _jacobianOplusXi[2] = -y;
  } 

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

public:
  double _x; // x value, y value is _measurement
};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;//actual parameter value
  double ae = 2.0, be = -1.0, ce = 5.0;//estimated parameter value
  int N = 100;//data point
  double w_sigma = 1.0;//Noise Sigma value
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;//OpenCV random number generator

  vector<double> x_data, y_data;//data
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }

  //Build graph optimization, first set g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType; // The optimization variable dimension of each error term is 3, and the error value dimension is 1
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type

  //Gradient descent method, you can choose from GN, LM, DogLeg
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;//graph model
  optimizer.setAlgorithm(solver);//set up the solver
  optimizer.setVerbose(true);//turn on debug output

  //Add vertices to the graph
  CurveFittingVertex *v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(ae, be, ce));
  v->setId(0);
  optimizer.addVertex(v);

  //Add edges to the graph
  for (int i = 0; i < N; i++) {
    CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    edge->setVertex(0, v);//set the connected vertices
    edge->setMeasurement(y_data[i]);//observation value
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));//information matrix: the inverse of the covariance matrix
    optimizer.addEdge(edge);
  }

  //perform optimization
  cout << "start optimization" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  //output optimized value
  Eigen::Vector3d abc_estimate = v->estimate();
  cout << "estimated model: " << abc_estimate.transpose() << endl;

  return 0;
}