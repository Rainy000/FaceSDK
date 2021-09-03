
#include <iostream>
#include <string>

#include "face_detector.h"
#include "net.h"
#include "face_sdk.h"
#include <opencv2/opencv.hpp>

int main()
{


	char* image_path_1 = "/home/wangy/workspace/FaceTest/image/000_0.bmp";
	char* image_path_2 = "/home/wangy/workspace/FaceTest/image/000_0.bmp";
    char* image_path_3 = "/home/wangy/workspace/FaceTest/image/multi-people.jpg";
	char* center_path = "/home/wangy/workspace/FaceTest/models/detect";
	char* faceModel_path = "/home/wangy/workspace/FaceTest/models/recognition";
	
	// initial 

	Centerface *Detector =  new Centerface;
	int detector_ret = Detector->init(center_path);
	if (detector_ret != 0)
	{
		std::cout << "Failed to load detector model." << std::endl;
		return -1;
	}
	
	
	int ret = initial(center_path, faceModel_path);
	if (ret==0)
	{
	    std::cout<<"model initialization success."<<std::endl;
	}
	
    cv::Mat image1 = cv::imread(image_path_1);
	std::vector<FaceInfo> face_vec1;
	ncnn::Mat inmat1 = ncnn::Mat::from_pixels(image1.data, ncnn::Mat::PIXEL_BGR2RGB, image1.cols, image1.rows);

	Detector->detect(inmat1, face_vec1, image1.cols, image1.rows);

    cv::Mat image2 = cv::imread(image_path_2);
	std::vector<FaceInfo> face_vec2;
	ncnn::Mat inmat2 = ncnn::Mat::from_pixels(image2.data, ncnn::Mat::PIXEL_BGR2RGB, image2.cols, image2.rows);

	Detector->detect(inmat2, face_vec2, image2.cols, image2.rows);    

	double d_start = (double)cvGetTickCount();
	auto result1 = getFeat(image_path_1, face_vec1[0].landmarks, 10);
	auto result2 = getFeat(image_path_2, face_vec2[0].landmarks, 10);
	double d_end = (double)cvGetTickCount();
	std::cout << "feat time: " << (d_end - d_start) / (cvGetTickFrequency() * 1000 * 2) << " ms " << std::endl;
	float sim = featCompare(result1, result2);
	std::cout << "score = " << sim << std::endl;
	
	
	int num1, num2;
	d_start = (double)cvGetTickCount();
	auto result1v = getDetectFeat(image_path_1, &num1);
	auto result2v = getDetectFeat(image_path_2, &num2);
	d_end = (double)cvGetTickCount();
	std::cout << "detect+feat time: " << (d_end - d_start) / (cvGetTickFrequency() * 1000 * 2) << " ms " << std::endl;
	float simv = featCompare(result1v, result2v);
	std::cout << "scorev = " << simv << std::endl;
	
    
    int num3;
    d_start = (double)cvGetTickCount();
    auto result3 = getAllDetectFeat(image_path_3, &num3);
    d_end = (double)cvGetTickCount();
	std::cout << "multi-people detect+feat time: " << (d_end - d_start) / (cvGetTickFrequency() * 1000 * 2) << " ms " << std::endl;
    std::cout<<"multi-people detect+feat num = "<< num3<<std::endl;
    if (num3 >=1)
    {
        
        for(int i =0;i<num3;i++)
        {
            std::cout<<"face index = "<< i <<" x1 = "<< result3[i].left<<" y1 = "<<result3[i].top<<" x2 = "<< result3[i].right << " y2= "<<result3[i].bottom<<std::endl;
            std::cout<<"feat size = "<<result3[i].FeatureSize<<std::endl;
        }
    }
    
	releaseFaceInfo(result1);
	releaseFaceInfo(result2);
	releaseFaceInfo(result1v);
    releaseFaceInfo(result2v);
	releaseArrayFaceInfo(result3);

    int model_ret = releaseModel();
    if (model_ret==0)
        std::cout<<"release model success."<<std::endl;
    if (Detector!=NULL )
    {
        delete Detector;
        Detector = nullptr;
    }

	return 0;
}
