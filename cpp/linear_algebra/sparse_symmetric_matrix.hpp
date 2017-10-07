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
  virtual const MatrixX<ScalarT> BlockAt(size_t row_id, size_t col_id) const override;
  virtual const MatrixX<ScalarT> operator()(size_t row_id, size_t col_id) const override { return BlockAt(row_id, col_id); }

  virtual FastBlockIndex EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) override;
  virtual void EmplaceZeroBlock(size_t row_id, size_t col_id);
  virtual void SetBlockZeroAt(size_t row_id, size_t col_id);
  virtual SparseVector<ScalarT> RemoveColAt(size_t col_id);
};


/*******************************************************************************
 * The implementation of SparseSymmetricMatrix<ScalarT>
 *******************************************************************************/

template<typename ScalarT>
const MatrixX<ScalarT> SparseSymmetricMatrix<ScalarT>::BlockAt(size_t row_id, size_t col_id) const {
  if (row_id <= col_id)
    return SparseMatrix<ScalarT>::BlockAt(row_id, col_id);
  else
    return SparseMatrix<ScalarT>::BlockAt(col_id, row_id).Transpose();
}

template<typename ScalarT>
FastBlockIndex SparseSymmetricMatrix<ScalarT>::EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) {
  FastBlockIndex block_id;
  if (row_id <= col_id) {
    block_id = SparseMatrix<ScalarT>::EmplaceBlock(block, row_id, col_id);
    this->row_vectors_[col_id][row_id] = block_id;
    this->col_blocks_[row_id][col_id] = block_id;
    this->rows_ = (col_id + 1) > this->rows_ ? (col_id + 1) : this->rows_;
    this->cols_ = (row_id + 1) > this->cols_ ? (row_id + 1) : this->cols_;
  } else {
    block_id = SparseMatrix<ScalarT>::EmplaceBlock(block.Transpose(), col_id, row_id);
    this->row_vectors_[row_id][col_id] = block_id;
    this->col_blocks_[col_id][row_id] = block_id;
    this->rows_ = (row_id + 1) > this->rows_ ? (row_id + 1) : this->rows_;
    this->cols_ = (col_id + 1) > this->cols_ ? (col_id + 1) : this->cols_;
  }
  return block_id;
}

template<typename ScalarT>
void SparseSymmetricMatrix<ScalarT>::EmplaceZeroBlock(size_t row_id, size_t col_id) {
  SparseMatrix<ScalarT>::EmplaceZeroBlock(row_id, col_id);
  SparseMatrix<ScalarT>::EmplaceZeroBlock(col_id, row_id);
}

template<typename ScalarT>
void SparseSymmetricMatrix<ScalarT>::SetBlockZeroAt(size_t row_id, size_t col_id) {
  SparseMatrix<ScalarT>::SetBlockZeroAt(row_id, col_id);
  SparseMatrix<ScalarT>::SetBlockZeroAt(col_id, row_id);
}

template<typename ScalarT>
SparseVector<ScalarT> SparseSymmetricMatrix<ScalarT>::RemoveColAt(size_t col_id) {
  SparseVector<ScalarT> col_vector = SparseMatrix<ScalarT>::RemoveColAt(col_id);

  // remove the row at col_id
  size_t row_id = col_id;
  auto row_entry = this->row_vectors_.find(row_id);
  if (row_entry != this->row_vectors_.end()) {
    for (auto block_entry : row_entry->second) {
      size_t col_id = block_entry->first;
      auto& col_vector = this->col_vectors_[col_id];
      for (size_t col = col_id; col < this->cols(); ++col) {
        auto next_block_entry = this->col_vector.find(col + 1);
        if (next_block_entry == this->col_vector.end())
          col_vector.erase(col);
        else
          col_vector[col] = next_block_entry->second;
      }
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

  return col_vector;
}

}
