
#pragma once
#include "net.h"
#include <opencv2/core/core.hpp>
#include <string>

#include <vector>

void print_feature(std::vector<float>& feature);
class FaceExtractor {
public:
	FaceExtractor();
	~FaceExtractor();
	int LoadModel(const std::string model_path, const std::string model_name);
	int ExtractFeature(cv::Mat &img, std::vector<float> *feature);

private:
	ncnn::Net *faceExt;
	bool initialized;

	const static int in_w;
	const static int in_h;
	const static int FeatureDim;
	const static int num_thread;
};

