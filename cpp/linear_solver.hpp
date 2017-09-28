#ifndef CERES_PRO_LINEAR_SOLVER_H_
#define CERES_PRO_LINEAR_SOLVER_H_

namespace ceres_pro {

class ProblemBase;

struct Options {
};

struct Solution {
};

class LinearSolver {
 public:
  LinearSolver(const Options& options) : options_(options) {}

  void Solve(ProblemBase& problem);

 protected:
  Options options_;
};

}

#endif  // CERES_PRO_LINEAR_SOLVER_H_
