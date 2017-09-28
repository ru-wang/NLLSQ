#ifndef CERES_PRO_SPECIAL_VIEW_PROBLEM_H_
#define CERES_PRO_SPECIAL_VIEW_PROBLEM_H_

#include "problem.hpp"

namespace ceres_pro {

/*
 * The conditional view for linear least
 * square problem class for problems of
 * the form: J'QJx = J'Qr,
 * where Q is the information matrix.
 */
template<typename ScalarT>
class ConditionalViewProblem : public LinearProblem<ScalarT> {
 public:
  ConditionalViewProblem(std::initializer_list<size_t> var_ids) {
  }
};

/*
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
