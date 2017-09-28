#ifndef CERES_PRO_LOSS_FUNCTOR_H_
#define CERES_PRO_LOSS_FUNCTOR_H_

#include "sparse_matrix.hpp"

namespace ceres_pro {

template<typename ScalarT, size_t VAR_NUM>
class LossFunctor {
 public:
  friend class ProblemBase;

  virtual ~LossFunctor() = default;

  /*
   * Compute the residuals and the corresponding Jacobians.
   * */
  virtual bool operator()(VectorX<ScalarT> const* const* variables,
                          MatrixX<ScalarT> const* information,
                          VectorX<ScalarT>* residuals,
                          MatrixX<ScalarT>* jacobians) const = 0;

 private:
  const VectorX<ScalarT>* variables_[VAR_NUM];
  const MatrixX<ScalarT>* information_;
};

}

#endif  // CERES_PRO_LOSS_FUNCTOR_H_
