#ifndef CERES_PRO_VECTOR_H_
#define CERES_PRO_VECTOR_H_

#include <Eigen/Eigen>

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>

namespace ceres_pro {
class FastBlockIndex {
 public:
  friend struct std::hash<FastBlockIndex>;
  FastBlockIndex(size_t id) : id_(id) {}
  static FastBlockIndex Gen() {
    static size_t id_counter = 0;
    return id_counter++;
  }
 private:
  size_t id_;
};
}

namespace std {
template<> struct hash<ceres_pro::FastBlockIndex> {
  size_t operator()(const ceres_pro::FastBlockIndex& s) const {
    return s.id_;
  }
};
}

namespace ceres_pro {

//template<typename ScalarT> using MatrixX = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ScalarT>
class MatrixX : public Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> {
 public:
  bool operator!() const { return this->rows() == this->cols() == 0; }
  operator bool() const { return operator!(); }
  static MatrixX<ScalarT> ZeroBlock() {
    static const Eigen::Matrix<ScalarT, 0, 0> zero;
    return zero;
  }
};

/*
 * Dense vector structure.
 *
 * All blocks are stored in column-major.
 */
template<typename ScalarT>
class DenseVector {
 public:
  const MatrixX<ScalarT>& BlockAt(size_t row_id) const;
        MatrixX<ScalarT>& BlockAt(size_t row_id);

  const MatrixX<ScalarT>& operator[](size_t row_id) const { return BlockAt(row_id); }
        MatrixX<ScalarT>& operator[](size_t row_id)       { return BlockAt(row_id); }

  FastBlockIndex PushBlockBack(const MatrixX<ScalarT>& block);
  void RemoveBlockAt(size_t row_id);

  size_t Rows() const { return blocks_.size(); }

 protected:
  void check_dimension(size_t cols);

  std::vector<MatrixX<ScalarT>> blocks_;
};

/*
 * Sparse vector structure.
 *
 * All blocks are stored in column-major.
 */
template<typename ScalarT>
class SparseVector {
 public:
  SparseVector() : rows_(0), cols_(0) {}

  const MatrixX<ScalarT>& BlockAt(size_t row_id) const;
        MatrixX<ScalarT>& BlockAt(size_t row_id);

  const MatrixX<ScalarT>& operator[](size_t row_id) const { return BlockAt(row_id); }
        MatrixX<ScalarT>& operator[](size_t row_id)       { return BlockAt(row_id); }

  FastBlockIndex AddBlock(const MatrixX<ScalarT>& block, size_t row_id);
  void RemoveBlockAt(size_t row_id);
  void SetBlockZeroAt(size_t row_id);

  size_t Rows() const { return rows_; }

 protected:
  void check_dimension(size_t cols);

  std::unordered_map<FastBlockIndex, MatrixX<ScalarT>> blocks_;
  std::unordered_map<size_t, FastBlockIndex> row_blocks_;

 private:
  size_t rows_;
  size_t cols_;
};


/*
 * The implementation of DenseVector<ScalarT>
 */

template<typename ScalarT>
const MatrixX<ScalarT>& DenseVector<ScalarT>::BlockAt(size_t row_id) const {
  CHECK(row_id >= 0 && row_id < Rows()) << "Dense vector index out of range!";
  return blocks_[row_id];
}

template<typename ScalarT>
MatrixX<ScalarT>& DenseVector<ScalarT>::BlockAt(size_t row_id) {
  return const_cast<MatrixX<ScalarT>&>(static_cast<const DenseVector<ScalarT>*>(this)->BlockAt(row_id));
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
void DenseVector<ScalarT>::RemoveBlockAt(size_t row_id) {
  CHECK(row_id >= 0 && row_id < Rows()) << "Dense vector index out of range!";
  blocks_.erase(row_id);
}

template<typename ScalarT>
void DenseVector<ScalarT>::check_dimension(size_t cols) {
  if (!blocks_.empty())
    CHECK_EQ(cols, blocks_.front().cols()) << "Wrong block size!";
}


/*
 * The implementation of SparseVector<ScalarT>
 */

template<typename ScalarT>
const MatrixX<ScalarT>& SparseVector<ScalarT>::BlockAt(size_t row_id) const {
  CHECK(row_id >= 0 && row_id < Rows()) << "Sparse vector index out of range!";
  auto entry = row_blocks_.find(row_id);
  if (entry != row_blocks_.end()) {
    FastBlockIndex block_id = entry->second;
    return blocks_[block_id];
  } else {
    return MatrixX<ScalarT>::ZeroBlock();
  }
}

template<typename ScalarT>
MatrixX<ScalarT>& SparseVector<ScalarT>::BlockAt(size_t row_id) {
  return const_cast<MatrixX<ScalarT>&>(static_cast<const SparseVector<ScalarT>*>(this)->BlockAt(row_id));
}

template<typename ScalarT>
FastBlockIndex SparseVector<ScalarT>::AddBlock(const MatrixX<ScalarT>& block, size_t row_id) {
  check_dimension(block->cols);
  FastBlockIndex block_id = FastBlockIndex::Gen();
  blocks_[block_id] = block;
  row_blocks_[row_id] = block_id;
  rows_ = (row_id + 1) > rows_ ? (row_id + 1) : rows_;
  return block_id;
}

template<typename ScalarT>
void SparseVector<ScalarT>::SetBlockZeroAt(size_t row_id) {
  auto entry = row_blocks_.find(row_id);
  if (entry != row_blocks_.end()) {
    FastBlockIndex block_id = entry->second;
    blocks_.erase(block_id);
  }
  rows_ = (row_id + 1) > rows_ ? (row_id + 1) : rows_;
}

template<typename ScalarT>
void SparseVector<ScalarT>::RemoveBlockAt(size_t row_id) {
  CHECK(row_id >= 0 && row_id < Rows()) << "Sparse vector index out of range!";
  auto entry = row_blocks_.find(row_id);

  if (entry != row_blocks_.end()) {
    FastBlockIndex block_id = entry->second;
    blocks_.erase(block_id);
  } else {
    LOG(ERROR) << "Empty block";
  }

  // update the adjacency list
  for (size_t row = row_id; row < rows_ - 1; ++row) {
    auto next_row = row_blocks_.find(row + 1);
    if (next_row == row_blocks_.end())
      row_blocks_.erase(row);
    else
      row_blocks_[row] = next_row->second;
  }

  --rows_;
}

template<typename ScalarT>
void SparseVector<ScalarT>::check_dimension(size_t cols) {
  if (cols_ == 0)
    cols_ = cols;
  else
    CHECK_EQ(cols, cols_) << "Wrong block size!";
}

}

#endif  // CERES_PRO_VECTOR_H_
