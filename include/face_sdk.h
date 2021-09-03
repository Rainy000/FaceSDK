#pragma once
#ifndef FACE_SDK_H
#define FACE_SDK_H
extern "C"
{
	struct Face_Info
	{
		int left;
		int top;
		int right;
		int bottom;
		int FeatureSize;
		float feature[256];
		Face_Info() {
			left = 0;
			top = 0;
			right = 0;
			bottom = 0;
			FeatureSize = 0;
			feature[256] = { 0 };
		};
        ~Face_Info(){};
	};
    /*struct FaceInfo_Array
    {
        Face_Info FaceArray[5];
    };
    */
	int initial(char *detectPath, char *featPath);
	Face_Info* getDetectFeat(char* imgPath, int* faceNum);
	//FaceInfo_Array* getAllDetectFeat(char* imgPath, int* faceNum);
	Face_Info* getAllDetectFeat(char* imgPath, int* faceNum);
	Face_Info* getFeat(char* imgPath, float landmarks[], int length);
	float featCompare(Face_Info* feat1, Face_Info* feat2);
	void releaseFaceInfo(Face_Info* feat);
	void releaseArrayFaceInfo(Face_Info* feat);
	int releaseModel();
}


#endif
