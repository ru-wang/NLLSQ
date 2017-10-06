#pragma once

#include <Eigen/Eigen>

#include <functional>

namespace ceres_pro {

/*******************************************************************************
 * Special block index for fast access.
 *******************************************************************************/
class FastBlockIndex {
 public:
  friend struct std::hash<FastBlockIndex>;
  FastBlockIndex() = default;
  FastBlockIndex(size_t id) : id_(id) {}
  static FastBlockIndex Gen() {
    static size_t id_counter = 0;
    return id_counter++;
  }
 private:
  size_t id_;
};

/*******************************************************************************
 * Dense block structure.
 *
 * All elements are stored in column-major.
 *******************************************************************************/
template<typename ScalarT>
class MatrixX : public Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> {
 public:
  MatrixX() { this->resize(0, 0); }
  operator bool() const { return this->rows() != 0 && this->cols() != 0; }
  bool operator!() const { return !(operator bool()); }
  static const MatrixX& ZeroBlock() {
    static const MatrixX zero;
    return zero;
  }
};

}

namespace std {

template<> struct hash<ceres_pro::FastBlockIndex> {
  size_t operator()(const ceres_pro::FastBlockIndex& s) const {
    return s.id_;
  }
};

}
