#ifndef CERES_PRO_NONLINEAR_SOLVER_H_
#define CERES_PRO_NONLINEAR_SOLVER_H_

#include "linear_solver.hpp"

namespace ceres_pro {

class NonlinearSolver : public SolverBase {
 public:
  virtual ~NonlinearSolver() override = default;

  virtual const Solution& Solve(ProblemBase* problem) override;
};

}

#endif  // CERES_PRO_NONLINEAR_SOLVER_H_
