#pragma once

#include "linear_algebra/matrixx.hpp"
#include "linear_algebra/sparse_vector.hpp"

#include <Eigen/Eigen>

#include <glog/logging.h>

#include <unordered_map>

namespace ceres_pro {

template<typename ScalarT> class SparseMatrix;

template<typename ScalarT>
std::ostream& operator<<(std::ostream& os, const SparseMatrix<ScalarT>& mat);


/*******************************************************************************
 * Sparse matrix structure.
 *
 * All blocks are stored in column-major.
 *******************************************************************************/
template<typename ScalarT>
class SparseMatrix {
  friend std::ostream& operator<<<>(std::ostream& os, const SparseMatrix& mat);

 public:
  SparseMatrix() : rows_(0), cols_(0) {}

  virtual MatrixX<ScalarT> BlockAt(size_t row_id, size_t col_id) const;
  virtual MatrixX<ScalarT> operator()(size_t row_id, size_t col_id) const { return BlockAt(row_id, col_id); }

  virtual FastBlockIndex EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id);
  virtual void EmplaceZeroBlock(size_t row_id, size_t col_id);
  virtual void SetBlockZeroAt(size_t row_id, size_t col_id);
  virtual SparseVector<ScalarT> RemoveColAt(size_t col_id);

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

 protected:
  void check_dimension(size_t rows, size_t cols, size_t row_id, size_t col_id);

  std::unordered_map<FastBlockIndex, MatrixX<ScalarT>> blocks_;
  std::unordered_map<size_t, std::unordered_map<size_t, FastBlockIndex>> row_vectors_;
  std::unordered_map<size_t, std::unordered_map<size_t, FastBlockIndex>> col_vectors_;

  size_t rows_;
  size_t cols_;
};


/*******************************************************************************
 * The implementation of SparseMatrix<ScalarT>
 *******************************************************************************/

template<typename ScalarT>
MatrixX<ScalarT> SparseMatrix<ScalarT>::BlockAt(size_t row_id, size_t col_id) const {
  CHECK(row_id >= 0 && row_id < rows()) << "Sparse matrix index out of range!";
  CHECK(col_id >= 0 && col_id < cols()) << "Sparse matrix index out of range!";
  auto row_entry = row_vectors_.find(row_id);
  if (row_entry != row_vectors_.end()) {
    auto block_entry = row_entry->second.find(col_id);
    if (block_entry != row_entry->second.end()) {
      FastBlockIndex block_id = block_entry->second;
      auto block = blocks_.find(block_id);
      CHECK(block != blocks_.end()) << "block[" << row_id << "," << col_id << "]";
      return block->second;
    }
  }
  return MatrixX<ScalarT>::ZeroBlock();
}

template<typename ScalarT>
FastBlockIndex SparseMatrix<ScalarT>::EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) {
  LOG(INFO) << "Emplacing block at \t(" << row_id << ", " << col_id << ")";

  CHECK(row_id >= 0) << "Sparse matrix index out of range!";
  CHECK(col_id >= 0) << "Sparse matrix index out of range!";
  check_dimension(block.rows(), block.cols(), row_id, col_id);

  FastBlockIndex block_id;
  auto row_entry = row_vectors_.find(row_id);
  if (row_entry != row_vectors_.end()) {
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
  LOG(INFO) << "Emplacing zero at \t(" << row_id << ", " << col_id << ")";

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

      auto block = blocks_.find(block_entry.second);
      CHECK(block != blocks_.end()) << "block (" << row_id << "," << col_id << ")";
      if (block->second) {
        FastBlockIndex block_id = block_entry.second;

        // construct the returned column vector
        col_vector.EmplaceBlock(block->second, row_id);

        // remove the blocks
        blocks_.erase(block_id);
      }
    }
  }

  // update the row adjacency list
  for (size_t row = 0; row < rows(); ++row) {
    auto& row_vector = row_vectors_[row];
    for (size_t col = col_id; col < cols(); ++col) {
      auto next_block_entry = row_vector.find(col + 1);
      if (next_block_entry == row_vector.end())
        row_vector.erase(col);
      else
        row_vector[col] = next_block_entry->second;
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
    CHECK(!row_entry->second.empty()) << "row " << row_id << " " << "col " << col_id << " "
                                      << "rows " << rows << " " << "cols " << cols;
    FastBlockIndex block_id = row_entry->second.cbegin()->second;
    auto block = blocks_.find(block_id);
    CHECK(block != blocks_.end()) << "block (" << row_id << "," << col_id << ")";
    size_t rows_at_row = block->second.rows();
    CHECK(rows_at_row == rows) << "Wrong block size!";
  }

  auto col_entry = col_vectors_.find(col_id);
  if (col_entry != col_vectors_.end()) {
    CHECK(!col_entry->second.empty()) << "row " << row_id << " " << "col " << col_id << " "
                                      << "rows " << rows << " " << "cols " << cols;
    FastBlockIndex block_id = col_entry->second.cbegin()->second;
    auto block = blocks_.find(block_id);
    CHECK(block != blocks_.end()) << "block (" << row_id << "," << col_id << ")";
    size_t cols_at_col = block->second.cols();
    CHECK(cols_at_col == cols) << "Wrong block size!";
  }
}

template<typename ScalarT>
std::ostream& operator<<(std::ostream& os, const SparseMatrix<ScalarT>& mat) {
  for (size_t row = 0; row < mat.rows(); ++row) {
    auto row_entry = mat.row_vectors_.find(row);

    if (row_entry != mat.row_vectors_.end()) {
      for (size_t col = 0; col < mat.cols(); ++col) {
        auto block_entry = row_entry->second.find(col);
        if (block_entry == row_entry->second.end()) {
          os << "O";
        } else {
          auto block = mat.blocks_.find(block_entry->second);
          CHECK(block != mat.blocks_.end()) << "block[" << row << "," << col << "]";
          os << block->second;
        }
        if (col < mat.cols() - 1)
          os << "\t";
      }
      if (row < mat.rows() - 1)
        os << "\n";
    } else {
      for (size_t col = 0; col < mat.cols(); ++col) {
        os << "O";
        if (col < mat.cols() - 1)
          os << "\t";
      }
      if (row < mat.rows() - 1)
        os << "\n";
    }
  }

  return os;
}

}
