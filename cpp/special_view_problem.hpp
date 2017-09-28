#ifndef CERES_PRO_SPECIAL_VIEW_PROBLEM_H_
#define CERES_PRO_SPECIAL_VIEW_PROBLEM_H_

#include "problem.hpp"

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
  ConditionalViewProblem(std::initializer_list<size_t> var_ids) {
    for (size_t variable_id : var_ids) {
      size_t col_id = variable_id;
      x_.RemoveBlockAt(var_ids);
      // TODO move to right-hand side
      A_.RemoveColAt(col_id);
    }
  }

 protected:
  using LinearProblem<ScalarT>::A_;
  using LinearProblem<ScalarT>::b_;
  using LinearProblem<ScalarT>::x_;
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
};

}

#endif  // CERES_PRO_SPECIAL_VIEW_PROBLEM_H_
