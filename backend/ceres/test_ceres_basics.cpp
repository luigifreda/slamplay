// A very simple example of using the Ceres minimizer.
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic, numeric and analytic differentiation. Compare run times.

#include "ceres/ceres.h"
#include "glog/logging.h"
#include <chrono>

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::SizedCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


// There are 3 ways of building a cost function 

// 1) Automatic differentiation. A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct TemplatedCostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

// 2) Numeric differentiation. A non-templated cost functor that implements the residual r = 10 -
// x. Case in which template cannot be used (for instance when recurring to an external lib)
struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
  residual[0] = 10.0 - x[0];
  return true;
  }
};


// 3) Analytic differentiation. A CostFunction implementing analytically derivatives for the
// function f(x) = 10 - x.
class AnalyticCostFunction
    : public SizedCostFunction<1 /* number of residuals */,
                               1 /* size of first parameter */> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    double x = parameters[0][0];
    // f(x) = 10 - x.
    residuals[0] = 10 - x;
    // f'(x) = -1. Since there's only 1 parameter and that parameter
    // has 1 dimension, there is only 1 element to fill in the
    // jacobians.
    //
    // Since the Evaluate function can be called with the jacobians
    // pointer equal to nullptr, the Evaluate function must check to see
    // if jacobians need to be computed.
    //
    // For this simple problem it is overkill to check if jacobians[0]
    // is nullptr, but in general when writing more complex
    // CostFunctions, it is possible that Ceres may only demand the
    // derivatives w.r.t. a subset of the parameter blocks.
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};


int mainAutoDiff(int argc, char** argv) 
{
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new AutoDiffCostFunction<TemplatedCostFunctor, 1, 1>(new TemplatedCostFunctor); // output size, input size
  problem.AddResidualBlock(cost_function, nullptr, &x);

  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";

  return 0;
}


int mainNumericDiff(int argc, char** argv) 
{
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses numeric diff. 
  CostFunction* cost_function =
    new NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(  // output size, input size
    new NumericDiffCostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);


  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";

  return 0;
}



int mainAnalyticDiff(int argc, char** argv) 
{
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual).
  CostFunction* cost_function = new AnalyticCostFunction;
  problem.AddResidualBlock(cost_function, nullptr, &x);

  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";

  return 0;
}


int main(int argc, char** argv) 
{
  google::InitGoogleLogging(argv[0]);

  std::chrono::steady_clock::time_point t1, t2; 
  std::chrono::duration<double> time_used;

  /*
  NOTE:
  Generally speaking we recommend automatic differentiation instead of numeric
  differentiation. The use of C++ templates makes automatic differentiation efficient, whereas
  numeric differentiation is expensive, prone to numeric errors, and leads to slower
  convergence.
  */

  std::cout << "=========================" << std::endl; 
  std::cout << " automatic differentiation (autodiff) " << std::endl; 
  t1 = std::chrono::steady_clock::now();
  mainAutoDiff(argc, argv); 
  t2 = std::chrono::steady_clock::now();
  time_used = std::chrono::duration_cast< std::chrono::duration<double >> (t2 - t1);
  std::cout << "duration:" << time_used.count() << " seconds" << std::endl;      

  std::cout << "=========================" << std::endl; 
  std::cout << " numeric differentiation " << std::endl;   
  t1 = std::chrono::steady_clock::now();
  mainNumericDiff(argc, argv); 
  t2 = std::chrono::steady_clock::now();
  time_used = std::chrono::duration_cast< std::chrono::duration<double >> (t2 - t1);
  std::cout << "duration:" << time_used.count() << " seconds" << std::endl; 

  std::cout << "=========================" << std::endl; 
  std::cout << " analaytic differentiation " << std::endl;   
  t1 = std::chrono::steady_clock::now();
  mainAnalyticDiff(argc, argv); 
  t2 = std::chrono::steady_clock::now();
  time_used = std::chrono::duration_cast< std::chrono::duration<double >> (t2 - t1);
  std::cout << "duration:" << time_used.count() << " seconds" << std::endl; 

  return 0; 
}