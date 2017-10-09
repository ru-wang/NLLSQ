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
  LogToStderr();

  MatrixX<short> b(1, 1);
  MatrixX<short> dmat = MatrixX<short>::RandomBlock(15, 15);
  SparseMatrix<short> smat;

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<short> r(0, 1);

  for (size_t row = 0; row < dmat.rows(); ++row) {
    for (size_t col = 0; col < dmat.cols(); ++col) {
      dmat(row, col) *= r(gen);
      if (dmat(row, col) != 0) {
        b(0, 0) = dmat(row, col);
        smat.EmplaceBlock(b, row, col);
      } else {
        smat.EmplaceZeroBlock(row, col);
      }
    }
  }

  LOG(INFO) << "random dense matrix\n";
  cout << "---" << dmat.rows() << "x" << dmat.cols() << "---\n"
       << dmat << "\n------\n";

  LOG(INFO) << "testing sparse matrix\n";
  cout << "---" << smat.rows() << "x" << smat.cols() << "---\n"
       << smat << "\n------\n";

  for (size_t row = 0; row < dmat.rows(); ++row) {
    for (size_t col = 0; col < dmat.cols(); ++col) {
      if (smat(row, col))
        CHECK(smat(row, col)(0, 0) == dmat(row, col)) << "(" << row << "," << col << ")";
      else
        CHECK(dmat(row, col) == 0) << "(" << row << "," << col << ")";
    }
  }

  LOG(INFO) << "testing sparse matrix (removing)\n";
  auto svec = smat.RemoveColAt(1);
  cout << "---" << svec.rows() << "x1" << "---\n"
       << svec << "\n------\n";

  svec = smat.RemoveColAt(5);
  cout << "---" << svec.rows() << "x1" << "---\n"
       << svec << "\n------\n";

  cout << "---" << smat.rows() << "x" << smat.cols() << "---\n"
       << smat << "\n------\n";

  return 0;
}
