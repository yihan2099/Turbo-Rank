# From deploy/
g++ infer.cpp -O2 -std=c++17 \
  -Iinclude \
  -Llib -lonnxruntime \
  -o infer

export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./infer