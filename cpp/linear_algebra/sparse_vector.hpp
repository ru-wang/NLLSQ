#pragma once

#include "linear_algebra/matrixx.hpp"

#include <Eigen/Eigen>

#include <glog/logging.h>

#include <ostream>
#include <unordered_map>

namespace ceres_pro {

template<typename ScalarT> class SparseVector;

template<typename ScalarT>
std::ostream& operator<<(std::ostream& os, const SparseVector<ScalarT>& vec);


/*******************************************************************************
 * Sparse vector structure.
 *
 * All blocks are stored in column-major.
 *******************************************************************************/
template<typename ScalarT>
class SparseVector {
  friend std::ostream& operator<<<>(std::ostream& os, const SparseVector&);

 public:
  SparseVector() : rows_(0) {}
  SparseVector(size_t rows) : rows_(rows) {}

  const MatrixX<ScalarT>& BlockAt(size_t row_id) const;
  const MatrixX<ScalarT>& operator[](size_t row_id) const { return BlockAt(row_id); }

  FastBlockIndex EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id);
  void EmplaceZeroBlock(size_t row_id);
  void SetBlockZeroAt(size_t row_id);
  MatrixX<ScalarT> RemoveBlockAt(size_t row_id);

  size_t rows() const { return rows_; }

 protected:
  void check_dimension(size_t cols);

  std::unordered_map<FastBlockIndex, MatrixX<ScalarT>> blocks_;
  std::unordered_map<size_t, FastBlockIndex> row_blocks_;

  size_t rows_;
};


/*******************************************************************************
 * The implementation of SparseVector<ScalarT>
 *******************************************************************************/

template<typename ScalarT>
const MatrixX<ScalarT>& SparseVector<ScalarT>::BlockAt(size_t row_id) const {
  CHECK(row_id >= 0 && row_id < rows()) << "Sparse vector index out of range!";
  auto entry = row_blocks_.find(row_id);
  if (entry != row_blocks_.end()) {
    FastBlockIndex block_id = entry->second;
    return blocks_.find(block_id)->second;
  } else {
    return MatrixX<ScalarT>::ZeroBlock();
  }
}

template<typename ScalarT>
FastBlockIndex SparseVector<ScalarT>::EmplaceBlock(const MatrixX<ScalarT>& block, size_t row_id) {
  if (!block)
    EmplaceZeroBlock(row_id);

  CHECK(row_id >= 0) << "Sparse vector index out of range!";
  check_dimension(block.cols());
  FastBlockIndex block_id;
  auto entry = row_blocks_.find(row_id);
  if (entry == row_blocks_.end())
    block_id = FastBlockIndex::Gen();
  else
    block_id = entry->second;
  blocks_[block_id] = block;
  row_blocks_[row_id] = block_id;
  rows_ = (row_id + 1) > rows_ ? (row_id + 1) : rows_;
  return block_id;
}

template<typename ScalarT>
void SparseVector<ScalarT>::EmplaceZeroBlock(size_t row_id) {
  CHECK(row_id >= 0) << "Sparse vector index out of range!";
  if (row_id >= rows())
    rows_ = (row_id + 1) > rows_ ? (row_id + 1) : rows_;
  else
    SetBlockZeroAt(row_id);
}

template<typename ScalarT>
void SparseVector<ScalarT>::SetBlockZeroAt(size_t row_id) {
  CHECK(row_id >= 0 && row_id < rows()) << "Sparse vector index out of range!";
  auto entry = row_blocks_.find(row_id);
  if (entry != row_blocks_.end()) {
    FastBlockIndex block_id = entry->second;
    blocks_.erase(block_id);
    row_blocks_.erase(entry);
  }
}

template<typename ScalarT>
MatrixX<ScalarT> SparseVector<ScalarT>::RemoveBlockAt(size_t row_id) {
  CHECK(row_id >= 0 && row_id < rows()) << "Sparse vector index out of range!";
  MatrixX<ScalarT> block = MatrixX<ScalarT>::ZeroBlock();
  auto block_entry = row_blocks_.find(row_id);
  if (block_entry != row_blocks_.end()) {
    FastBlockIndex block_id = block_entry->second;
    block = blocks_[block_id];
    blocks_.erase(block_id);
  }
  for (size_t row = row_id; row < rows(); ++row) {
    auto next_block_entry = row_blocks_.find(row + 1);
    if (next_block_entry == row_blocks_.end())
      row_blocks_.erase(row);
    else
      row_blocks_[row] = next_block_entry->second;
  }
  --rows_;
  return block;
}

template<typename ScalarT>
void SparseVector<ScalarT>::check_dimension(size_t cols) {
  CHECK(cols != 0) << "Wrong block size!";
  if (!blocks_.empty())
    CHECK(cols == blocks_.cbegin()->second.cols()) << "Wrong block size!";
}

template<typename ScalarT>
std::ostream& operator<<(std::ostream& os, const SparseVector<ScalarT>& vec) {
  size_t rows = 0, cols = 0;
  if (!vec.blocks_.empty()) {
    rows = vec.blocks_.cbegin()->second.rows();
    cols = vec.blocks_.cbegin()->second.cols();
  }
  for (size_t row = 0; row < vec.rows(); ++row) {
    auto block_entry = vec.row_blocks_.find(row);
    if (block_entry == vec.row_blocks_.end())
      os << "O";
    else
      os << vec.blocks_.find(block_entry->second)->second;
    if (row < vec.rows() - 1)
      os << "\n";
  }
  return os;
}

}
