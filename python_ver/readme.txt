how to build, in this folder
--------
git clone https://github.com/pybind/pybind11.git

mkdir build
cd build
conda activate base
cmake ..
cmake --build . --config Release
cd ..
git clone https://github.com/zeke-xie/adaptive-inertia-adai.git

Then copy Bortfeld.*.pyd or Bortfeld.*.so to adaptive-inertia-adai folder
copy test.py to adaptive-inertia-adai folder
run test.py
