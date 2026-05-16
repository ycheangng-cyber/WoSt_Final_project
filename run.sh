# 1. Enable the modern compiler
source /opt/rh/devtoolset-11/enable

# 2. (Optional but recommended) Source OneAPI
source /opt/intel/oneapi/setvars.sh

# 3. Clean and Build
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)

cd ..

build/wost
