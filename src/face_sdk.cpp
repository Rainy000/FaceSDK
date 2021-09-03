
#include <string>
#include <iostream>
#include <vector>
#include "net.h"
#include "face_sdk.h"
#include "face_detector.h"
#include "face_feature.h"
#include <opencv2/opencv.hpp>

Centerface *Detector = nullptr;
FaceExtractor *Extractor = nullptr;

float std_points[10] = // 96*112, 7 background
{
	30.2946, 51.6963,
	65.5318, 51.5014,
	48.0252, 71.7366,
	33.5493, 92.3655,
	62.7299, 92.2041,
};


///---------------------------------------------------------------------///
void CalcTransParam(float* facial_points, float* std_points, float* trans_param, int dst_width, int  dst_height, int type)
{
	const int NUM_FACIAL_POINT = 5;

	if (0 == type)
	{
		float x_ratio = dst_width / 96.0f;
		float y_ratio = dst_height / 112.0f;
		for (int i = 0; i < 5; i++)
		{
			std_points[2 * i] = std_points[2 * i] * x_ratio;
			std_points[2 * i + 1] = std_points[2 * i + 1] * y_ratio;
		}
	}
	else if (1 == type)
	{
		float offset_x = dst_width - 96;
		float offset_y = dst_height - 112;
		for (int i = 0; i < 5; i++)
		{
			std_points[2 * i] = std_points[2 * i] + offset_x / 2;
			std_points[2 * i + 1] = std_points[2 * i + 1] + offset_y;
		}
	}

	float sum_x = 0, sum_y = 0;
	float sum_u = 0, sum_v = 0;
	float sum_xx_yy = 0;
	float sum_ux_vy = 0;
	float sum_vx__uy = 0;
	for (int cnt = 0; cnt < NUM_FACIAL_POINT; ++cnt)
	{
		int x_off = cnt * 2;
		int y_off = x_off + 1;
		sum_x += std_points[x_off];
		sum_y += std_points[y_off];
		sum_u += facial_points[x_off];
		sum_v += facial_points[y_off];
		sum_xx_yy += std_points[x_off] * std_points[x_off] +
			std_points[y_off] * std_points[y_off];
		sum_ux_vy += std_points[x_off] * facial_points[x_off] +
			std_points[y_off] * facial_points[y_off];
		sum_vx__uy += facial_points[y_off] * std_points[x_off] -
			facial_points[x_off] * std_points[y_off];
	}

	float q = sum_u - sum_x * sum_ux_vy / sum_xx_yy
		+ sum_y * sum_vx__uy / sum_xx_yy;

	float p = sum_v - sum_y * sum_ux_vy / sum_xx_yy
		- sum_x * sum_vx__uy / sum_xx_yy;

	float r = NUM_FACIAL_POINT - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy;

	float a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy;

	float b = (sum_vx__uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy;

	float c = q / r;

	float d = p / r;

	trans_param[0] = a;
	trans_param[1] = -b;
	trans_param[2] = c;
	trans_param[3] = b;
	trans_param[4] = a;
	trans_param[5] = d;

	return;
}


unsigned char Sampling(const unsigned char* const feat_map,
	int c, int H, int W, int C, double x, double y, double scale) {
	// bilinear subsampling
	int ux = floor(x), uy = floor(y);
	double ans = 0;
	if (ux >= 0 && ux < H - 1 && uy >= 0 && uy < W - 1) {
		int offset = (ux * W + uy) * C + c;

		double cof_x = x - ux;
		double cof_y = y - uy;
		ans = (1 - cof_y) * feat_map[offset] + cof_y * feat_map[offset + C];
		ans = (1 - cof_x) * ans + cof_x * ((1 - cof_y) * feat_map[offset + W * C]
			+ cof_y * feat_map[offset + W * C + C]);
	}
	return ans;
}


void TransformImage(cv::Mat src_im, cv::Mat &dst_im,
	float *facial_points, float* std_points,
	int dst_width, int  dst_height, int dst_channels, int type, float shrink)
{

	//dst_im.data = new unsigned char[dst_width*dst_height*dst_channels];
	float trans_param[6];
	CalcTransParam(facial_points, std_points, trans_param, dst_width, dst_height, type);

	unsigned char* output_data = dst_im.data;
	unsigned char* input_data = src_im.data;

	double scale = sqrt(trans_param[0] * trans_param[0]
		+ trans_param[3] * trans_param[3]);
	for (int x = 0; x < dst_height; ++x) {
		for (int y = 0; y < dst_width; ++y) {
			// Get the source position of each point on the destination feature map.
			double src_y = trans_param[0] * y + trans_param[1] * x + trans_param[2];
			double src_x = trans_param[3] * y + trans_param[4] * x + trans_param[5];
			for (int c = 0; c < dst_channels; ++c) {
				output_data[x*dst_width*dst_channels + y * dst_channels + c]
					= Sampling(input_data, c, src_im.rows, src_im.cols,
						src_im.channels(), src_x, src_y, 1.0 / scale);
				/*cout << (int)output_data[x*dst_channels] << "  ";
				cin.get();*/
			}
		}
	}
	//imwrite(path_dst, img_norm);
}

float getMold(const std::vector<float>& vec){ 
    int n = vec.size();
    float sum = 0.0;
    for (int i = 0; i<n; ++i)
        sum += vec[i] * vec[i];
    return sqrt(sum);
}

float getSimilarity(const std::vector<float>& lhs, const std::vector<float>& rhs){
    //std::cout<<"lhs mod = "<<getMold(lhs)<<std::endl;
    //std::cout<<"rhs mod = "<<getMold(rhs)<<std::endl;
    int n = lhs.size();
    assert(n == rhs.size());
    float tmp = 0.0;  //内积
    for (int i = 0; i<n; ++i)
        tmp += lhs[i] * rhs[i];
    return tmp / (getMold(lhs)*getMold(rhs));
}

int initial(char *detectPath, char *featPath)
{

	// detector
	std::string DetectPath = detectPath;
	Detector = new Centerface;
	int detector_ret = Detector->init(DetectPath);
	if (detector_ret != 0)
	{
		std::cout << "Failed to load detector model." << std::endl;
		return -1;
	}
	// feature
	std::string FeaturePath = featPath;
	std::string faceModel_name = "model";
	Extractor = new FaceExtractor;
	int ret = Extractor->LoadModel(FeaturePath, faceModel_name);
	if (ret != 0) {
		std::cout << "Failed to load FaceID model." << std::endl;
		return -2;
	}
	return 0;
}

Face_Info* getFeat(char* imgPath, float landmarks[], int length)
{
	std::string image_path = imgPath;
	cv::Mat image = cv::imread(image_path);
	if (image.data == nullptr)
	{
		std::cout << "image is empty, please check." << std::endl;
		return nullptr;
	}
	else
	{
		Face_Info *face_ret = new Face_Info;

		cv::Mat croppedFace(112, 96, CV_8UC3);
		float facial_points[10] = { 0.0 };
		for (int i = 0; i < 5; i++)
		{
			if ((2 * i + 1) < length)
			{
				facial_points[2 * i] = static_cast<float>(landmarks[2 * i]);
				facial_points[2 * i + 1] = static_cast<float>(landmarks[2 * i + 1]);
			}
		}

		TransformImage(image, croppedFace, facial_points, std_points, 96, 112, 3, 1, 1.0f);
		cv::resize(croppedFace, croppedFace, cv::Size(112, 112));
		cv::Mat croppedFace_flip;
		cv::flip(croppedFace, croppedFace_flip, 1);
		// get feature
		std::vector<float> feature;
		std::vector<float> feature_flip;

		Extractor->ExtractFeature(croppedFace, &feature);
		Extractor->ExtractFeature(croppedFace_flip, &feature_flip);
		// feature concate
		face_ret->FeatureSize = (int)(feature.size());
		for (int i = 0; i < feature.size(); i++)
			face_ret->feature[i] = feature[i] + feature_flip[i];
		return face_ret;
	}
}

Face_Info* getDetectFeat(char* imgPath, int* faceNum)
{
	std::string image_path = imgPath;

	cv::Mat image = cv::imread(image_path);
	if (image.data == nullptr)
	{
		std::cout << "image is empty, please check." << std::endl;
		return nullptr;
	}
	else
	{
		std::vector<FaceInfo> face_vec;
		ncnn::Mat inmat = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
		Detector->detect(inmat, face_vec, image.cols, image.rows);
		//
		*faceNum = (int)(face_vec.size());
		if (face_vec.size() == 0)
		{
			std::cout << "detector result is nothing." << std::endl;
			return nullptr;
		}
		else
		{
			Face_Info *face_ret = new Face_Info;
			// find max face 
			FaceInfo face;
			if (face_vec.size() == 1)
			{
				face = face_vec[0];
			}
			else
			{ 
				int max_area = 0;
				int max_index = -1;
				for (int i = 0; i < face_vec.size(); i++) {
					auto temp_face = face_vec[i];
					int width = temp_face.x2 - temp_face.x1 + 1;
					int height = temp_face.y2 - temp_face.y1 + 1;
					if (width*height > max_index)
						max_index = i;
				}
				face = face_vec[max_index];
			}
			cv::Mat croppedFace(112, 96, CV_8UC3);
			float facial_points[10] = { 0.0 };
			for (int i = 0; i < 5; i++)
			{
				facial_points[2 * i] = static_cast<float>(face.landmarks[2 * i]);
				facial_points[2 * i + 1] = static_cast<float>(face.landmarks[2 * i + 1]);
			}
			TransformImage(image, croppedFace, facial_points, std_points, 96, 112, 3, 1, 1.0f);
			cv::resize(croppedFace, croppedFace, cv::Size(112, 112));
			cv::Mat croppedFace_flip;
			cv::flip(croppedFace, croppedFace_flip, 1);
			// get feature
			std::vector<float> feature;
			std::vector<float> feature_flip;

			Extractor->ExtractFeature(croppedFace, &feature);
			Extractor->ExtractFeature(croppedFace_flip, &feature_flip);
			// feature concate
			face_ret->left = face.x1;
			face_ret->top = face.y1;
			face_ret->right = face.x2;
			face_ret->bottom = face.y2;
			face_ret->FeatureSize = (int)(feature.size());
			for (int i = 0; i < feature.size(); i++)
				face_ret->feature[i] = feature[i] + feature_flip[i];
			return face_ret;
		}
	}
}


Face_Info* getAllDetectFeat(char* imgPath, int* faceNum)
{
    double d_start = (double)cvGetTickCount();
	std::string image_path = imgPath;

	cv::Mat image = cv::imread(image_path);
	if (image.data == nullptr)
	{
		std::cout << "image is empty, please check." << std::endl;
		return nullptr;
	}
	else
	{
		std::vector<FaceInfo> face_vec;
		ncnn::Mat inmat = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
		Detector->detect(inmat, face_vec, image.cols, image.rows);
		//
		
		
		if (face_vec.size() == 0)
		{
			std::cout << "detector result is nothing." << std::endl;
            *faceNum = 0;
			return nullptr;
		}
		else
		{

            Face_Info *face_ret = new Face_Info[int(face_vec.size())];
            Face_Info *temp = face_ret;
			// calculate each face feat 
			int valid_num = 0;
			for (int i = 0; i < face_vec.size(); i++) {
				auto face = face_vec[i];
				int width = face.x2 - face.x1 + 1;
				int height = face.y2 - face.y1 + 1;
				if (width<40 || height<40)
					continue;
				cv::Mat croppedFace(112, 96, CV_8UC3);
				float facial_points[10] = { 0.0 };
				for (int i = 0; i < 5; i++){
					facial_points[2 * i] = static_cast<float>(face.landmarks[2 * i]);
					facial_points[2 * i + 1] = static_cast<float>(face.landmarks[2 * i + 1]);
                    //cv::circle(image, cv::Point(facial_points[2*i],facial_points[2*i+1]),3,(255,0,0),2);
				}
                //cv::rectangle(image, cv::Point(face.x1,face.y1), cv::Point(face.x2,face.y2), cv::Scalar(255,0,0),1,1,0);
				TransformImage(image, croppedFace, facial_points, std_points, 96, 112, 3, 1, 1.0f);
				cv::resize(croppedFace, croppedFace, cv::Size(112, 112));
				cv::Mat croppedFace_flip;
				cv::flip(croppedFace, croppedFace_flip, 1);
				// get feature
				std::vector<float> feature;
				std::vector<float> feature_flip;

				Extractor->ExtractFeature(croppedFace, &feature);
				Extractor->ExtractFeature(croppedFace_flip, &feature_flip);
				// feature concate
				temp->left = face.x1;
				temp->top = face.y1;
				temp->right = face.x2;
				temp->bottom = face.y2;
				temp->FeatureSize = (int)(feature.size());
				for (int j = 0; j < feature.size(); j++)
					temp->feature[j] = feature[j] + feature_flip[j];
				valid_num +=1;
                temp = temp+1;
			}
            *faceNum = valid_num;
            double d_end = (double)cvGetTickCount();
	        std::cout << "cost time: " << (d_end - d_start) / (cvGetTickFrequency() * 1000 * 2) << " ms " << std::endl;
			return face_ret;
		}
	}
}



void releaseFaceInfo(Face_Info* feat)
{
	if (feat != nullptr)
	{
		delete feat;
		return;
	}
	else
	{
		std::cout << "input faceinfo is null." << std::endl;
		return;
	}
}

void releaseArrayFaceInfo(Face_Info* feat)
{
	if(feat != nullptr)
	{
		delete []feat;
		return;
	}
	else
	{
		std::cout<<"input faceinfo is null."<<std::endl;
		return;
	}
}

float featCompare(Face_Info* feat1, Face_Info* feat2)
{
    std::vector<float> lf(feat1->feature, feat1->feature+256);
    std::vector<float> rf(feat2->feature, feat2->feature+256);
	float score = getSimilarity(lf, rf);
	return score;
}

int releaseModel()
{
    if (Detector!=NULL )
    {
        delete Detector;
        Detector = nullptr;
    }
    if (Extractor!=NULL)
    {
        delete Extractor;
        Extractor = nullptr;
    }
    return 0;
}
