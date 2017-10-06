#pragma once

#include "matrix"

#include <array>

namespace ceres_pro {

template<typename ScalarT, size_t VAR_NUM>
class LossFunctor {
 public:
  friend class ProblemBase;

  virtual ~LossFunctor() = default;

  /*
   * Compute the residuals and the corresponding Jacobians.
   * */
  virtual bool operator()(MatrixX<ScalarT> const* const* variables,
                          MatrixX<ScalarT> const* information,
                          MatrixX<ScalarT>* residuals,
                          MatrixX<ScalarT>* jacobians) const = 0;

 private:
  std::array<const MatrixX<ScalarT>*, VAR_NUM> variables_;
  const MatrixX<ScalarT>* information_;
};

}
