#pragma once

#include "linear_algebra/matrixx.hpp"

#include <Eigen/Eigen>

#include <glog/logging.h>

#include <vector>

namespace ceres_pro {

/*******************************************************************************
 * Dense vector structure.
 *
 * All blocks are stored in column-major.
 *******************************************************************************/
template<typename ScalarT>
class DenseVector {
 public:
  DenseVector() = default;
  DenseVector(size_t rows) : blocks_(rows) {}

  const MatrixX<ScalarT>& BlockAt(size_t row_id) const;
  const MatrixX<ScalarT>& operator[](size_t row_id) const { return BlockAt(row_id); }

  FastBlockIndex PushBlockBack(const MatrixX<ScalarT>& block);
  void SetBlockZeroAt(size_t row_id);
  void RemoveBlockAt(size_t row_id);

  size_t rows() const { return blocks_.size(); }

 protected:
  void check_dimension(size_t cols);

  std::vector<MatrixX<ScalarT>> blocks_;
};


/*******************************************************************************
 * The implementation of DenseVector<ScalarT>
 *******************************************************************************/

template<typename ScalarT>
const MatrixX<ScalarT>& DenseVector<ScalarT>::BlockAt(size_t row_id) const {
  CHECK(row_id >= 0 && row_id < rows()) << "Dense vector index out of range!";
  return blocks_[row_id];
}

template<typename ScalarT>
FastBlockIndex DenseVector<ScalarT>::PushBlockBack(const MatrixX<ScalarT>& block) {
  check_dimension(block->cols);
  size_t row_id = blocks_.size();
  FastBlockIndex block_id = row_id;
  blocks_.push_back(block);
  return block_id;
}

template<typename ScalarT>
void DenseVector<ScalarT>::SetBlockZeroAt(size_t row_id) {
  CHECK(row_id >= 0 && row_id < rows()) << "Dense vector index out of range!";
  blocks_[row_id] = MatrixX<ScalarT>::ZeroBlock();
}

template<typename ScalarT>
void DenseVector<ScalarT>::RemoveBlockAt(size_t row_id) {
  CHECK(row_id >= 0 && row_id < rows()) << "Dense vector index out of range!";
  blocks_.erase(row_id);
}

template<typename ScalarT>
void DenseVector<ScalarT>::check_dimension(size_t cols) {
  if (!blocks_.empty())
    CHECK_EQ(cols, blocks_.front().cols()) << "Wrong block size!";
}

}
