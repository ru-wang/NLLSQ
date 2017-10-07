#include "matrix"

#include <glog/logging.h>

#include <iostream>

using namespace ceres_pro;
using namespace google;
using namespace std;

int main(int /*argc*/, char* argv[]) {
  InitGoogleLogging(argv[0]);
  LogToStderr();

  MatrixX<int> b;
  SparseVector<int> svec;

  LOG(INFO) << "testing sparse vector\n";
  b = MatrixX<int>::RandomBlock(3, 1); svec.EmplaceBlock(b,   0);
  b = MatrixX<int>::RandomBlock(3, 1); svec.EmplaceBlock(b,   2);
                                       svec.EmplaceZeroBlock( 4);
  b = MatrixX<int>::RandomBlock(3, 1); svec.EmplaceBlock(b,   6);
                                       svec.EmplaceZeroBlock(10);
  cout << "---" << svec.rows() << "x1" << "---\n"
       << svec << "\n------\n";

  LOG(INFO) << "testing sparse vector (removing)\n";
  svec.RemoveBlockAt(0);
  svec.RemoveBlockAt(0);
  svec.RemoveBlockAt(0);
  svec.EmplaceBlock(svec[3], 5);
  cout << "---" << svec.rows() << "x1" << "---\n"
       << svec << "\n------\n";

  LOG(INFO) << "testing sparse vector (setting zero)\n";
  svec.EmplaceZeroBlock(10);
  svec.SetBlockZeroAt(5);
  cout << "---" << svec.rows() << "x1" << "---\n"
       << svec << "\n------\n";

  return 0;
}
