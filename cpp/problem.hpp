#pragma once

#include "loss_functor.hpp"
#include "matrix"

#include <glog/logging.h>

#include <initializer_list>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace ceres_pro {

/*
 * The abstract linear problem class.
 */
class ProblemBase {
 public:
  virtual ~ProblemBase() {}
};

/*
 * The linear problem class for problems
 * of the form: Ax = b.
 */
template<typename ScalarT>
class LinearProblem : public ProblemBase {
 public:
  virtual ~LinearProblem() override = default;

 protected:
  SparseMatrix<ScalarT> A_;
  DenseVector<ScalarT> b_;
  DenseVector<ScalarT> x_;
};

/*
 * The linear least square problem class
 * for problems of the form: J'QJx = J'Qr,
 * where Q is the information matrix.
 */
template<typename ScalarT>
class QuadraticProblem : public LinearProblem<ScalarT> {
 public:

  virtual ~QuadraticProblem() override = default;

  size_t AddVariableBlock(MatrixX<ScalarT>* variable) {
    size_t variable_id = this->x_.PushBlockBack(variable);
    problem_graph_[variable_id] = std::make_pair(variable_id, std::vector<size_t>());
    return variable_id;
  }

  template<size_t RES_DIM>
  size_t AddLossFunctor(LossFunctor<ScalarT, RES_DIM>* loss_functor,
                        MatrixX<ScalarT>* info,
                        std::initializer_list<size_t> var_ids) {
    size_t residual_id = loss_functors_.size();
    loss_functors_.push_back(loss_functor);
    loss_map_[loss_functors_] = residual_id;

    size_t i = 0;
    for (size_t variable_id : var_ids)
      loss_functor->variables_[i++] = this->x_[variable_id];
    loss_functor->information_ = info;

    for (size_t variable_id : var_ids)
      problem_graph_[variable_id].push_back(residual_id);

    for (size_t variable_id : var_ids)
      this->J_(residual_id, variable_id).resize(RES_DIM, this->x_[variable_id].Rows());
    this->b_[residual_id].resize(RES_DIM);
    this->Q_(residual_id, residual_id) = info;

    return residual_id;
  }

 protected:
  std::vector<void*> loss_functors_;
  std::unordered_map<void*, size_t> loss_map_;

  /*
   * [ variable_id ] --> [ loss_0, loss_1, ..., loss_n ]
   */
  std::unordered_map<size_t, std::vector<size_t>> problem_graph_;

  SparseMatrix<ScalarT> J_;
  SparseMatrix<ScalarT> Q_;
};

}

#endif  // CERES_PRO_PROBLEM_H_
