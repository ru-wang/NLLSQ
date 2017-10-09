#include "matrix"

#include <glog/logging.h>

#include <cmath>
#include <ctime>
#include <iostream>
#include <random>

using namespace ceres_pro;
using namespace google;
using namespace std;

int main(int /*argc*/, char* argv[]) {
  InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = 2;

  MatrixX<short> b(1, 1);
  MatrixX<short> dmat = MatrixX<short>::RandomBlock(15, 15);
  SparseSymmetricMatrix<short> ssmat;

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<short> r(0, 1);

  for (size_t row = 0; row < dmat.rows(); ++row)
    for (size_t col = 0; col < dmat.cols(); ++col)
      if ((dmat(row, col)) == 0)
        dmat(row, col) = 1;

  LOG(INFO) << "random dense matrix\n";
  cout << "---" << dmat.rows() << "x" << dmat.cols() << "---\n"
       << dmat << "\n------\n";

  for (size_t row = 0; row < dmat.rows(); ++row) {
    for (size_t col = 0; col < dmat.cols(); ++col) {
      if (dmat(row, col) != 0) {
        b(0, 0) = dmat(row, col);
        ssmat.EmplaceBlock(b, row, col);
      } else {
        ssmat.EmplaceZeroBlock(row, col);
      }
    }
  }

  LOG(INFO) << "testing sparse matrix\n";
  cout << "---" << ssmat.rows() << "x" << ssmat.cols() << "---\n"
       << ssmat << "\n------\n";

  for (size_t row = 0; row < dmat.rows(); ++row) {
    for (size_t col = 0; col < dmat.cols(); ++col) {
      auto block = ssmat(row, col);
      if (ssmat(row, col))
        CHECK_EQ(ssmat(row, col)(0, 0), dmat(row, col)) << "at (" << row << "," << col << ")";
      else
        CHECK_EQ(dmat(row, col), 0) << "at (" << row << "," << col << ")";
    }
  }

  return 0;
}
