#ifndef CERES_PRO_LINEAR_SOLVER_H_
#define CERES_PRO_LINEAR_SOLVER_H_

namespace ceres_pro {

class ProblemBase;

struct Options {
};

struct Solution {
  Solution() : usable(false) {}

  bool usable;
};

class SolverBase {
 public:
  SolverBase(Options options) : options_(options) {}

  virtual ~SolverBase() = default;

  virtual const Solution& Solve(ProblemBase* problem) = 0;

  const Options& options() const { return options_; }
  const Solution& solution() const { return solution_; }

  void set_options(Options options) { options_ = options; }

 protected:
  Options options_;
  Solution solution_;
};

class LinearSolver : public SolverBase {
 public:
  LinearSolver(const Options& options) : SolverBase(options) {}

  virtual const Solution& Solve(ProblemBase* problem) override {
    return solution_;
  }
};

}

#endif  // CERES_PRO_LINEAR_SOLVER_H_
