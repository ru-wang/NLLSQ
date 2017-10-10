#pragma once

#include "linear_algebra/matrixx.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/sparse_vector.hpp"

#include <glog/logging.h>

namespace ceres_pro {


/*******************************************************************************
 * Sparse symmetric matrix structure.
 *
 * Only stores the upper triangular blocks,
 * and all blocks are stored in column-major.
 *******************************************************************************/
template<typename ScalarT>
class SparseSymmetricMatrix : public SparseMatrix<ScalarT> {
 public:
  virtual MatrixX<ScalarT> BlockAt(size_t row_id, size_t col_id) const override;
  virtual MatrixX<ScalarT> operator()(size_t row_id, size_t col_id) const override { return BlockAt(row_id, col_id); }

  virtual FastBlockIndex EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) override;
  virtual void EmplaceZeroBlock(size_t row_id, size_t col_id);
  virtual void SetBlockZeroAt(size_t row_id, size_t col_id);
  virtual SparseVector<ScalarT> RemoveColAt(size_t col_id);
};


/*******************************************************************************
 * The implementation of SparseSymmetricMatrix<ScalarT>
 *******************************************************************************/

template<typename ScalarT>
MatrixX<ScalarT> SparseSymmetricMatrix<ScalarT>::BlockAt(size_t row_id, size_t col_id) const {
  if (row_id <= col_id)
    return SparseMatrix<ScalarT>::BlockAt(row_id, col_id);
  else
    return SparseMatrix<ScalarT>::BlockAt(col_id, row_id).Transpose();
}

template<typename ScalarT>
FastBlockIndex SparseSymmetricMatrix<ScalarT>::EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) {
  FastBlockIndex block_id;
  if (row_id < col_id) {
    block_id = SparseMatrix<ScalarT>::EmplaceBlock(block, row_id, col_id);
    this->row_vectors_[col_id][row_id] = block_id;
    this->col_vectors_[row_id][col_id] = block_id;
    this->rows_ = (col_id + 1) > this->rows_ ? (col_id + 1) : this->rows_;
    this->cols_ = (row_id + 1) > this->cols_ ? (row_id + 1) : this->cols_;
  } else if (row_id > col_id) {
    block_id = SparseMatrix<ScalarT>::EmplaceBlock(block.Transpose(), col_id, row_id);
    this->row_vectors_[row_id][col_id] = block_id;
    this->col_vectors_[col_id][row_id] = block_id;
    this->rows_ = (row_id + 1) > this->rows_ ? (row_id + 1) : this->rows_;
    this->cols_ = (col_id + 1) > this->cols_ ? (col_id + 1) : this->cols_;
  } else {
    block_id = SparseMatrix<ScalarT>::EmplaceBlock(block, row_id, col_id);
  }
  return block_id;
}

template<typename ScalarT>
void SparseSymmetricMatrix<ScalarT>::EmplaceZeroBlock(size_t row_id, size_t col_id) {
  SparseMatrix<ScalarT>::EmplaceZeroBlock(row_id, col_id);
  if (row_id != col_id) {
    LOG(INFO) << "Emplacing zero at \t(" << col_id << ", " << row_id << ")";
    if (row_id >= this->rows() || col_id >= this->cols()) {
      this->rows_ = (col_id + 1) > this->rows_ ? (col_id + 1) : this->rows_;
      this->cols_ = (row_id + 1) > this->cols_ ? (row_id + 1) : this->cols_;
    } else {
      auto row_entry = this->row_vectors_.find(col_id);
      if (row_entry != this->row_vectors_.end()) {
        auto col_entry = row_entry->second.find(row_id);
        if (col_entry != row_entry->second.end()) {
          row_entry->second.erase(col_entry);
          this->col_vectors_[row_id].erase(col_id);
        }
      }
    }
  }
}

template<typename ScalarT>
void SparseSymmetricMatrix<ScalarT>::SetBlockZeroAt(size_t row_id, size_t col_id) {
  SparseMatrix<ScalarT>::SetBlockZeroAt(row_id, col_id);
  if (row_id != col_id) {
    CHECK(col_id >= 0 && col_id < this->rows()) << "Sparse matrix index out of range!";
    CHECK(row_id >= 0 && row_id < this->cols()) << "Sparse matrix index out of range!";
    auto row_entry = this->row_vectors_.find(col_id);
    if (row_entry != this->row_vectors_.end()) {
      auto col_entry = row_entry->second.find(row_id);
      if (col_entry != row_entry->second.end()) {
        row_entry->second.erase(col_entry);
        this->col_vectors_[row_id].erase(col_id);
      }
    }
  }
}

template<typename ScalarT>
SparseVector<ScalarT> SparseSymmetricMatrix<ScalarT>::RemoveColAt(size_t col_id) {
  SparseVector<ScalarT> col_vector = SparseMatrix<ScalarT>::RemoveColAt(col_id);

  // remove the row at col_id
  size_t row_id = col_id;

  // update the column adjacency list
  for (auto& col_entry : this->col_vectors_) {
    for (size_t row = row_id; row < this->rows(); ++row) {
      auto next_block_entry = col_entry.second.find(row + 1);
      if (next_block_entry == col_entry.second.end())
        col_entry.second.erase(row);
      else
        col_entry.second[row] = next_block_entry->second;
    }
  }

  // update the row adjacency list
  for (size_t row = row_id; row < this->rows(); ++row) {
    auto next_row_entry = this->row_vectors_.find(row + 1);
    if (next_row_entry == this->row_vectors_.end())
      this->row_vectors_.erase(row);
    else
      this->row_vectors_[row] = next_row_entry->second;
  }

  --this->rows_;

  return col_vector;
}

}
