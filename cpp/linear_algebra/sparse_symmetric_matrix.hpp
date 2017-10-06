#pragma once

#include "linear_algebra/matrixx.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/sparse_vector.hpp"

namespace ceres_pro {

/*
 * TODO
 * Sparse symmetric matrix structure.
 *
 * Only stores the upper triangular blocks,
 * and all blocks are stored in column-major.
 */
template<typename ScalarT>
class SparseSymmetricMatrix : public SparseMatrix<ScalarT> {
 public:
  virtual const MatrixX<ScalarT>& BlockAt(size_t row_id, size_t col_id) const override;
  virtual const MatrixX<ScalarT>& operator()(size_t row_id, size_t col_id) const override { return BlockAt(row_id, col_id); }
  virtual FastBlockIndex AddBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) override;
};


/*
 * TODO
 * The implementation of SparseSymmetricMatrix<ScalarT>
 */

template<typename ScalarT>
const MatrixX<ScalarT>& SparseSymmetricMatrix<ScalarT>::BlockAt(size_t row_id, size_t col_id) const {
  CHECK_NE(this->row_blocks_.find(row_id), this->row_blocks_.end()) << "Sparse symmetric matrix index out of range!";
  CHECK_NE(this->col_blocks_.find(col_id), this->col_blocks_.end()) << "Sparse symmetric matrix index out of range!";
  if (row_id <= col_id)
    return this->row_blocks_[row_id][col_id];
  else
    return this->row_blocks_[col_id][row_id].transpose();
}

template<typename ScalarT>
FastBlockIndex SparseSymmetricMatrix<ScalarT>::AddBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) {
  FastBlockIndex block_id = SparseMatrix<ScalarT>::AddBlock(block, row_id, col_id);
  this->row_blocks_[col_id][row_id] = block_id;
  this->col_blocks_[row_id][col_id] = block_id;
  this->rows_ = (row_id + 1) > this->rows_ ? (row_id + 1) : this->rows_;
  this->cols_ = (col_id + 1) > this->cols_ ? (col_id + 1) : this->cols_;
  return block_id;
}

}
