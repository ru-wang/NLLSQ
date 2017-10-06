#pragma once

#include "linear_solver.hpp"

namespace ceres_pro {

class NonlinearSolver : public SolverBase {
 public:
  virtual ~NonlinearSolver() override = default;

  virtual const Solution& Solve(ProblemBase* problem) override;
};

}
