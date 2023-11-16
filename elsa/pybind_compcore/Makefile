CXX := g++
CXXFLAGS := -std=c++14 -fPIC -shared -O3
INCLUDES := -I /home/liy/new/myenv/include/python3.11 -I/home/liy/packages/pybind11/include
LDFLAGS := -L /home/liy/new/myenv/lib
LDLIBS := -lpython3.11

SRC_FILES := $(wildcard ./*.cpp)
OBJ_FILES := $(patsubst %.cpp, %.o, $(SRC_FILES))

TARGET := ../lsa/compcore.so

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LDLIBS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ_FILES) $(TARGET)
