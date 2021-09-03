
rm -rf build
mkdir build
cd build

cmake -DOpenCV_DIR=/home/wangy/3rdParty/opencv348_sdk/share/OpenCV ..
make -j8

