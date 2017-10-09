CXX = g++ $(CXXFLAGS)
CXXFLAGS += -I.. \
						-isystem/usr/include/eigen3 \
						-std=c++11 \
						-g
LIBS = -lglog

OBJECTS = svec smat ssmat

all: $(OBJECTS)

svec: ./sparse_vector.cxx
	$(CXX) ./sparse_vector.cxx -o svec $(LIBS)

smat: ./sparse_matrix.cxx
	$(CXX) ./sparse_matrix.cxx -o smat $(LIBS)

ssmat: ./sparse_symmetric_matrix.cxx
	$(CXX) ./sparse_symmetric_matrix.cxx -o ssmat $(LIBS)

clean:
	rm -f $(OBJECTS)
