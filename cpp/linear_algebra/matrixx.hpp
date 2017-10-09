#pragma once

#include <Eigen/Eigen>

#include <functional>

namespace ceres_pro {

template<typename ScalarT> class MatrixX;

template<typename ScalarT>
std::ostream& operator<<(std::ostream& os, const MatrixX<ScalarT>& block);


/*******************************************************************************
 * Special block index for fast access.
 *******************************************************************************/
class FastBlockIndex {
  friend struct std::hash<FastBlockIndex>;
  friend std::ostream& operator<<(std::ostream& os, const FastBlockIndex& block);

 public:
  FastBlockIndex() = default;

  FastBlockIndex(size_t id) : id_(id) {}

  bool operator==(const FastBlockIndex& other) const {
    return id_ == other.id_;
  }

  static FastBlockIndex Gen() {
    static size_t id_counter = 0;
    return id_counter++;
  }

 private:
  size_t id_;
};

std::ostream& operator<<(std::ostream& os, const FastBlockIndex& block) {
  return os << block.id_;
}


/*******************************************************************************
 * Dense block structure.
 *
 * All elements are stored in column-major.
 *******************************************************************************/
template<typename ScalarT>
class MatrixX : private Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> {
  using EigenMat = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>;
  friend std::ostream& operator<<<>(std::ostream& os, const MatrixX& block);

 public:
  MatrixX() : EigenMat(0, 0), transposed_(false), is_zero_(true) {}
  MatrixX(size_t rows, size_t cols) : EigenMat(rows, cols), transposed_(false), is_zero_(false) {}
  MatrixX(const EigenMat& eigen_mat) : EigenMat(eigen_mat), transposed_(false), is_zero_(false) {}

  operator bool() const {
    return !is_zero_;
  }

  static const MatrixX RandomBlock(size_t rows, size_t cols) {
    return MatrixX(EigenMat::Random(rows, cols));
  }

  static const MatrixX ZeroBlock(size_t rows, size_t cols) {
    return MatrixX(rows, cols);
  }

  static const MatrixX& ZeroBlock() {
    static const MatrixX zero;
    return zero;
  }

  MatrixX Transpose() const {
    MatrixX transposed(*this);
    transposed.transposed_ = true^transposed_;
    return transposed;
  }

  MatrixX& TransposeInplace() {
    transposed_ ^= true;
    return (*this);
  }

  void SetZero() {
    is_zero_ = true;
  }

  bool operator==(const MatrixX& other) const {
    if (rows() != other.rows() || cols() != other.cols())
      return false;
    if (is_zero_ == other.is_zero_ == true)
      return true;
    for (size_t row = 0; row < rows(); ++row)
      for (size_t col = 0; col < cols(); ++col)
        if ((*this)(row, col) != other(row, col))
          return false;
    return true;
  }

  const ScalarT& operator()(size_t row, size_t col) const {
    return EigenMat::operator()(row, col);
  }

  ScalarT& operator()(size_t row, size_t col) {
    return EigenMat::operator()(row, col);
  }

  size_t rows() const { return EigenMat::rows(); }
  size_t cols() const { return EigenMat::cols(); }

  const EigenMat& KernalMat() const {
    return *static_cast<const EigenMat*>(this);
  };

  EigenMat& KernalMat() {
    return const_cast<EigenMat&>(static_cast<const MatrixX*>(this)->KernalMat());
  };

 private:
  bool transposed_;
  bool is_zero_;
};

template<typename ScalarT>
std::ostream& operator<<(std::ostream& os, const MatrixX<ScalarT>& block) {
  return os << static_cast<const Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>&>(block);
}

}

namespace std {

template<> struct hash<ceres_pro::FastBlockIndex> {
  size_t operator()(const ceres_pro::FastBlockIndex& s) const {
    return s.id_;
  }
};

}
