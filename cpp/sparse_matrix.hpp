#ifndef CERES_PRO_SPARSE_MATRIX_H_
#define CERES_PRO_SPARSE_MATRIX_H_

#include <Eigen/Eigen>

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <utility>

namespace ceres_pro {

template<typename ScalarT> using VectorX = Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>;
template<typename ScalarT> using MatrixX = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>;

/*
 * Sparse vector structure.
 *
 * All blocks are stored in column-major.
 */
template<typename ScalarT>
class DenseVector {
 public:
  VectorX<ScalarT>& BlockAt(size_t row_id) {
    CHECK(row_id >= 0 && row_id < Rows()) << "Dense vector index out of range!";
    return blocks_[row_id];
  }

  const VectorX<ScalarT>& BlockAt(size_t row_id) const {
    CHECK(row_id >= 0 && row_id < Rows()) << "Dense vector index out of range!";
    return blocks_[row_id];
  }

  virtual VectorX<ScalarT>& operator[](size_t row_id) { return BlockAt(row_id); }
  virtual VectorX<ScalarT>& operator()(size_t row_id) { return BlockAt(row_id); }

  virtual const VectorX<ScalarT>& operator[](size_t row_id) const { return BlockAt(row_id); }
  virtual const VectorX<ScalarT>& operator()(size_t row_id) const { return BlockAt(row_id); }

  virtual size_t PushBlockBack(VectorX<ScalarT>* block) {
    check_dimension(block->cols);
    size_t row_id = Rows();
    blocks_.push_back(block);
    return row_id;
  }

  size_t Rows() const { return blocks_.size(); }

 protected:
  std::vector<VectorX<ScalarT>> blocks_;

 private:
  void check_dimension(size_t cols) { CHECK_EQ(cols, 1) << "Wrong block size!"; }
};

/*
 * Sparse matrix structure.
 *
 * All blocks are stored in column-major.
 */
template<typename ScalarT>
class SparseMatrix {
 public:
  MatrixX<ScalarT>& BlockAt(size_t row_id, size_t col_id) {
    CHECK_NE(row_major_blocks_.find(row_id), row_major_blocks_.cend()) << "Sparse matrix index out of range!";
    CHECK_NE(col_major_blocks_.find(col_id), col_major_blocks_.cend()) << "Sparse matrix index out of range!";
    return row_major_blocks_[row_id][col_id];
  }

  const MatrixX<ScalarT>& BlockAt(size_t row_id, size_t col_id) const {
    CHECK_NE(row_major_blocks_.find(row_id), row_major_blocks_.cend()) << "Sparse matrix index out of range!";
    CHECK_NE(col_major_blocks_.find(col_id), col_major_blocks_.cend()) << "Sparse matrix index out of range!";
    return row_major_blocks_[row_id][col_id];
  }

  virtual MatrixX<ScalarT>& operator()(size_t row_id, size_t col_id) { return BlockAt(row_id, col_id); }
  virtual const MatrixX<ScalarT>& operator()(size_t row_id, size_t col_id) const { return BlockAt(row_id, col_id); }

  virtual size_t AddBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) {
    check_dimension(block.rows(), block.cols, row_id, col_id);
    size_t block_id = blocks_.size();
    blocks_.push_back(block);
    row_major_blocks_[row_id][col_id] = block_id;
    col_major_blocks_[col_id][row_id] = block_id;
    return block_id;
  }

  size_t Rows() const { row_major_blocks_.size(); }
  size_t Cols() const { col_major_blocks_.size(); }

 protected:
  std::vector<MatrixX<ScalarT>> blocks_;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> row_major_blocks_;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> col_major_blocks_;

 private:
  void check_dimension(size_t rows, size_t cols, size_t row_id, size_t col_id) {
    auto entry = row_major_blocks_.find(row_id);
    if (entry != row_major_blocks_.cend()) {
      size_t block_id = entry->second.cbegin()->second;
      size_t rows_at_row = blocks_[block_id].rows();
      CHECK_EQ(rows_at_row, rows) << "Wrong block size!";
    }

    entry = col_major_blocks_.find(col_id);
    if (entry != col_major_blocks_.cend()) {
      size_t block_id = entry->second.cbegin()->second;
      size_t cols_at_col = blocks_[block_id].cols();
      CHECK_EQ(cols_at_col, cols) << "Wrong block size!";
    }
  }
};

/*
 * Sparse symmetric matrix structure.
 *
 * Only stores the upper triangular blocks,
 * and all blocks are stored in column-major.
 */
template<typename ScalarT>
class SparseSymmetricMatrix : public SparseMatrix<ScalarT> {
 public:
  virtual const MatrixX<ScalarT>& BlockAt(size_t row_id, size_t col_id) const override {
    CHECK_NE(row_major_blocks_.find(row_id), row_major_blocks_.cend()) << "Sparse symmetric matrix index out of range!";
    CHECK_NE(col_major_blocks_.find(col_id), col_major_blocks_.cend()) << "Sparse symmetric matrix index out of range!";
    if (row_id <= col_id)
      return row_major_blocks_[row_id][col_id];
    else
      return row_major_blocks_[col_id][row_id].transpose();
  }

  virtual const MatrixX<ScalarT>& operator()(size_t row_id, size_t col_id) const override {
    return BlockAt(row_id, col_id);
  }

  virtual size_t AddBlock(const MatrixX<ScalarT>& block, size_t row_id, size_t col_id) override {
    size_t block_id = SparseMatrix<ScalarT>::AddBlock(block, row_id, col_id);
    row_major_blocks_[col_id][row_id] = block_id;
    col_major_blocks_[row_id][col_id] = block_id;
    return block_id;
  }

 protected:
  using SparseMatrix<ScalarT>::blocks_;
  using SparseMatrix<ScalarT>::row_major_blocks_;
  using SparseMatrix<ScalarT>::col_major_blocks_;
};

}

#endif  // CERES_PRO_SPARSE_MATRIX_H_
