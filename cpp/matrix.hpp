#ifndef CERES_PRO_MATRIX_H_
#define CERES_PRO_MATRIX_H_

#include "vector.hpp"

#include <Eigen/Eigen>

#include <glog/logging.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace ceres_pro {

/*
 * Sparse matrix structure.
 *
 * All blocks are stored in column-major.
 */
template<typename ScalarT>
class SparseMatrix {
 public:
  SparseMatrix() : rows_(0), cols_(0) {}

  virtual const MatrixX<ScalarT>& BlockAt(size_t row_id, size_t col_id) const;
  virtual       MatrixX<ScalarT>& BlockAt(size_t row_id, size_t col_id);
  virtual const MatrixX<ScalarT>& operator()(size_t row_id, size_t col_id) const { return BlockAt(row_id, col_id); }
  virtual       MatrixX<ScalarT>& operator()(size_t row_id, size_t col_id)       { return BlockAt(row_id, col_id); }

  virtual const SparseVector<ScalarT> ColAt(size_t col_id) const;
  virtual const SparseVector<ScalarT> operator[](size_t col_id) const { return ColAt(col_id); }

  virtual FastBlockIndex AddBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id);
  virtual void RemoveColAt(size_t col_id);
  virtual void SetBlockZeroAt(size_t row_id, size_t col_id);

  size_t Rows() const { return rows_; }
  size_t Cols() const { return cols_; }

 protected:
  std::vector<MatrixX<ScalarT>> blocks_;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> row_blocks_;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> col_blocks_;

 private:
  void check_dimension(size_t rows, size_t cols, size_t row_id, size_t col_id);

  size_t rows_;
  size_t cols_;
};

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
 * The implementation of SparseMatrix<ScalarT>
 */

template<typename ScalarT>
const MatrixX<ScalarT>& SparseMatrix<ScalarT>::BlockAt(size_t row_id, size_t col_id) const {
  CHECK_NE(row_blocks_.find(row_id), row_blocks_.end()) << "Sparse matrix index out of range!";
  CHECK_NE(col_blocks_.find(col_id), col_blocks_.end()) << "Sparse matrix index out of range!";
  FastBlockIndex block_id = row_blocks_[row_id][col_id];
  return blocks_[block_id];
}

template<typename ScalarT>
MatrixX<ScalarT>& SparseMatrix<ScalarT>::BlockAt(size_t row_id, size_t col_id) {
  return const_cast<MatrixX<ScalarT>&>(static_cast<const SparseMatrix<ScalarT>*>(this)->BlockAt(row_id, col_id));
}

template<typename ScalarT>
const SparseVector<ScalarT> SparseMatrix<ScalarT>::ColAt(size_t col_id) const {
}

template<typename ScalarT>
FastBlockIndex SparseMatrix<ScalarT>::AddBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) {
  check_dimension(block->blocks_.size(), block->cols, row_id, col_id);
  FastBlockIndex block_id = FastBlockIndex::Gen();
  blocks_[block_id] = block;
  row_blocks_[row_id][col_id] = block_id;
  col_blocks_[col_id][row_id] = block_id;
  rows_ = (row_id + 1) > rows_ ? (row_id + 1) : rows_;
  cols_ = (col_id + 1) > cols_ ? (col_id + 1) : cols_;
  return block_id;
}

template<typename ScalarT>
void SparseMatrix<ScalarT>::RemoveColAt(size_t col_id) {
  auto col_it = col_blocks_.find(col_id);
  CHECK_NE(col_it, col_blocks_.end()) << "Sparse matrix index out of range!";

  SparseVector<ScalarT> col_vector;

  // update the adjacency list
  for (size_t row_id = 0; row_id < rows_; ++row_id) {
  }

  col_blocks_.erase(col_it);
  --cols_;
}

// TODO
template<typename ScalarT>
void SparseMatrix<ScalarT>::SetBlockZeroAt(size_t row_id, size_t col_id) {
}

template<typename ScalarT>
void SparseMatrix<ScalarT>::check_dimension(size_t rows, size_t cols, size_t row_id, size_t col_id) {
  auto entry = row_blocks_.find(row_id);
  if (entry != row_blocks_.end()) {
    FastBlockIndex block_id = entry->second.cbegin()->second;
    size_t rows_at_row = blocks_[block_id].rows();
    CHECK_EQ(rows_at_row, rows) << "Wrong block size!";
  }

  entry = col_blocks_.find(col_id);
  if (entry != col_blocks_.end()) {
    FastBlockIndex block_id = entry->second.cbegin()->second;
    size_t cols_at_col = blocks_[block_id].cols();
    CHECK_EQ(cols_at_col, cols) << "Wrong block size!";
  }
}


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

#endif  // CERES_PRO_MATRIX_H_
