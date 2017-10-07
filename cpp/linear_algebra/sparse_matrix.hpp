#pragma once

#include "linear_algebra/matrixx.hpp"
#include "linear_algebra/sparse_vector.hpp"

#include <Eigen/Eigen>

#include <glog/logging.h>

#include <unordered_map>

namespace ceres_pro {

/*******************************************************************************
 * Sparse matrix structure.
 *
 * All blocks are stored in column-major.
 *******************************************************************************/
template<typename ScalarT>
class SparseMatrix {
 public:
  SparseMatrix() : rows_(0), cols_(0) {}

  virtual const MatrixX<ScalarT>& BlockAt(size_t row_id, size_t col_id) const;
  virtual const MatrixX<ScalarT>& operator()(size_t row_id, size_t col_id) const { return BlockAt(row_id, col_id); }

  virtual FastBlockIndex EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id);
  virtual void EmplaceZeroBlock(size_t row_id, size_t col_id);
  virtual void SetBlockZeroAt(size_t row_id, size_t col_id);
  virtual SparseVector<ScalarT> RemoveColAt(size_t col_id);

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

 protected:
  std::unordered_map<size_t, MatrixX<ScalarT>> blocks_;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> row_vectors_;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> col_vectors_;

 private:
  void check_dimension(size_t rows, size_t cols, size_t row_id, size_t col_id);

  size_t rows_;
  size_t cols_;
};


/*******************************************************************************
 * The implementation of SparseMatrix<ScalarT>
 *******************************************************************************/

template<typename ScalarT>
const MatrixX<ScalarT>& SparseMatrix<ScalarT>::BlockAt(size_t row_id, size_t col_id) const {
  CHECK(row_id >= 0 && row_id < rows()) << "Sparse matrix index out of range!";
  CHECK(col_id >= 0 && col_id < cols()) << "Sparse matrix index out of range!";
  auto row_entry = row_vectors_.find(row_id);
  if (row_entry != row_vectors_.end()) {
    auto block_entry = row_entry->second.find(col_id);
    if (block_entry != row_entry->second.end()) {
      FastBlockIndex block_id = block_entry->second;
      return blocks_[block_id];
    }
  }
  return MatrixX<ScalarT>::ZeroBlock();
}

template<typename ScalarT>
FastBlockIndex SparseMatrix<ScalarT>::EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) {
  if (!block)
    EmplaceZeroBlock(row_id, col_id);

  CHECK(row_id >= 0) << "Sparse matrix index out of range!";
  CHECK(col_id >= 0) << "Sparse matrix index out of range!";
  check_dimension(block.rows(), block.cols(), row_id, col_id);

  FastBlockIndex block_id;
  auto row_entry = row_vectors_.find(row_id);
  if (row_entry == row_vectors_.end()) {
    auto col_entry = row_entry->second.find(col_id);
    if (col_entry == row_entry->second.end())
      block_id = FastBlockIndex::Gen();
    else
      block_id = col_entry->second;
  } else {
    block_id = FastBlockIndex::Gen();
  }

  blocks_[block_id] = block;
  row_vectors_[row_id][col_id] = block_id;
  col_vectors_[col_id][row_id] = block_id;
  rows_ = (row_id + 1) > rows_ ? (row_id + 1) : rows_;
  cols_ = (col_id + 1) > cols_ ? (col_id + 1) : cols_;
  return block_id;
}

template<typename ScalarT>
void SparseMatrix<ScalarT>::EmplaceZeroBlock(size_t row_id, size_t col_id) {
  CHECK(row_id >= 0) << "Sparse matrix index out of range!";
  CHECK(col_id >= 0) << "Sparse matrix index out of range!";
  if (row_id >= rows() || col_id >= cols()) {
    rows_ = (row_id + 1) > rows_ ? (row_id + 1) : rows_;
    cols_ = (col_id + 1) > cols_ ? (col_id + 1) : cols_;
  } else {
    SetBlockZeroAt(row_id, col_id);
  }
}

template<typename ScalarT>
void SparseMatrix<ScalarT>::SetBlockZeroAt(size_t row_id, size_t col_id) {
  CHECK(row_id >= 0 && row_id < rows()) << "Sparse matrix index out of range!";
  CHECK(col_id >= 0 && col_id < cols()) << "Sparse matrix index out of range!";
  auto row_entry = row_vectors_.find(row_id);
  if (row_entry != row_vectors_.end()) {
    auto col_entry = row_entry->second.find(col_id);
    if (col_entry != row_entry->second.end()) {
      FastBlockIndex block_id = col_entry->second;
      blocks_.erase(block_id);
      row_entry->second.erase(col_entry);
      col_vectors_[col_id].erase(row_id);
    }
  }
}

template<typename ScalarT>
SparseVector<ScalarT> SparseMatrix<ScalarT>::RemoveColAt(size_t col_id) {
  CHECK(col_id >= 0 && col_id < cols()) << "Sparse matrix index out of range!";
  SparseVector<ScalarT> col_vector(rows());

  auto col_entry = col_vectors_.find(col_id);
  if (col_entry != col_vectors_.end()) {
    for (auto block_entry : col_entry->second) {
      size_t row_id = block_entry.first;

      if (block_entry.second) {
        FastBlockIndex block_id = block_entry.second;

        // construct the returned column vector
        col_vector.EmplaceBlock(blocks_[block_id], row_id);

        // remove the blocks
        blocks_.erase(block_id);
      }

      // update the row adjacency list
      auto& row_vector = row_vectors_[row_id];
      for (size_t col = col_id; col < cols(); ++col) {
        auto next_block_entry = row_vector.find(col + 1);
        if (next_block_entry == row_vector.end())
          row_vector.erase(col);
        else
          row_vector[col] = next_block_entry->second;
      }
    }
  }

  // update the column adjacency list
  for (size_t col = col_id; col < cols(); ++col) {
    auto next_col_entry = col_vectors_.find(col + 1);
    if (next_col_entry == col_vectors_.end())
      col_vectors_.erase(col);
    else
      col_vectors_[col] = next_col_entry->second;
  }

  --cols_;
  return col_vector;
}

template<typename ScalarT>
void SparseMatrix<ScalarT>::check_dimension(size_t rows, size_t cols, size_t row_id, size_t col_id) {
  auto row_entry = row_vectors_.find(row_id);
  if (row_entry != row_vectors_.end()) {
    FastBlockIndex block_id = row_entry->second.cbegin()->second;
    size_t rows_at_row = blocks_[block_id].rows();
    CHECK_EQ(rows_at_row, rows) << "Wrong block size!";
  }

  auto col_entry = col_vectors_.find(col_id);
  if (col_entry != col_vectors_.end()) {
    FastBlockIndex block_id = col_entry->second.cbegin()->second;
    size_t cols_at_col = blocks_[block_id].cols();
    CHECK_EQ(cols_at_col, cols) << "Wrong block size!";
  }
}

}
