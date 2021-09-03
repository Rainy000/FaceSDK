

#include "face_feature.h"
#include<iostream>
#include <string>
#if MIRROR_VULKAN
#include "gpu.h"
#endif //MIRROR_VULKAN

const int FaceExtractor::in_w = 112;
const int FaceExtractor::in_h = 112;
const int FaceExtractor::FeatureDim = 256;
const int FaceExtractor::num_thread = 4;

FaceExtractor::FaceExtractor()
{
	faceExt = new ncnn::Net();
	initialized = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();
	faceExt->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN

}

FaceExtractor::~FaceExtractor()
{
	faceExt->clear();
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN
}

void print_feature(std::vector<float>& feature)
{
    std::cout<<"Feature size = "<< feature.size()<<std::endl;
    for(int iter = 0; iter< feature.size(); iter++)
    {
        std::cout<< feature[iter]<<" ";
    }
    std::cout<<"\n";
    return;
}
int FaceExtractor::LoadModel(const std::string model_path, const std::string model_name)
{
	std::string param_path = model_path + "/" + model_name + ".param";
	std::string bin_path = model_path + "/" + model_name + ".bin";
	if (faceExt->load_param(param_path.c_str()) == -1 ||
		faceExt->load_model(bin_path.c_str()) == -1)
	{
		std::cout << "Load face recognition model failed." << std::endl;
		return -1;
	}
	initialized = true;
	return 0;

}
int FaceExtractor::ExtractFeature(cv::Mat &img, std::vector<float> *feature)
{
	//std::cout << "start extract feature." << std::endl;
	feature->clear();
	if (!initialized)
	{
		std::cout << "face recognition model uninitialized." << std::endl;
		return -1;
	}
	if (img.empty())
	{
		std::cout << "input image is empty." << std::endl;
		return -2;
	}

	cv::Mat img_cpy = img.clone();
	ncnn::Mat net_in = ncnn::Mat::from_pixels_resize(img_cpy.data, ncnn::Mat::PIXEL_BGR2RGB, img_cpy.cols, img_cpy.rows, in_w, in_h);
	feature->resize(FeatureDim);
	ncnn::Extractor ex = faceExt->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(num_thread);
	ex.input("data", net_in);
	ncnn::Mat net_out;
	ex.extract("fc1", net_out);
	for (int i = 0; i < FeatureDim; ++i)
	{
		feature->at(i) = net_out[i];
	}
    //std::cout<<"before NORM_L2, feature:"<<std::endl;
    //print_feature(*feature);
    //cv::normalize(*feature, *feature, 1, 0, cv::NORM_L2);
    //std::cout<<"after NORM_L2, feature:"<<std::endl;
    //print_feature(*feature);
	//std::cout << "End extract feature." << std::endl;
	return 0;
}
