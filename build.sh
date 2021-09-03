
SO_PATH=/home/wangy/workspace/project/FaceRecognition/lib/libglodonfacesdk.so

rm -f ${SO_PATH}
echo "Finish to delete old *.os file."

rm -rf build  && mkdir build && cd build
cmake -DOpenCV_DIR=/home/wangy/3rdParty/opencv348_sdk/share/OpenCV ..
make -j8

