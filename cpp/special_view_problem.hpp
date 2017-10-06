#pragma once

#include "problem.hpp"

#include <initializer_list>

namespace ceres_pro {

/*
 * TODO
 * The conditional view for linear least
 * square problem class for problems of
 * the form: J'QJx = J'Qr,
 * where Q is the information matrix.
 */
template<typename ScalarT>
class ConditionalViewProblem : public LinearProblem<ScalarT> {
 public:
  ConditionalViewProblem();

  ConditionalViewProblem& operator()(std::initializer_list<size_t> var_ids) {
    for (auto variable_id : var_ids) {
      this->A_ = original_problem_->A_;
    }
  }

 protected:
  LinearProblem<ScalarT>* original_problem_;
};

/*
 * TODO
 * The conditional view for linear least
 * square problem class for problems of
 * the form: J'QJx = J'Qr,
 * where Q is the information matrix.
 */
template<typename ScalarT>
class MarginalViewProblem : public LinearProblem<ScalarT> {
 public:
  MarginalViewProblem();
  MarginalViewProblem& operator()(std::initializer_list<size_t> var_ids);

 protected:
  LinearProblem<ScalarT>* original_problem_;
};

}
