#include "opencv2/opencv.hpp"
#include "yolov8-pose.h"
#include "chrono"


const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255}};

int main(int argc, char** argv)
{
    cudaSetDevice(0);
    auto yolov8_pose = new YOLOv8_pose("/home/pose/model/yolov8s-pose.engine");
    yolov8_pose->make_pipe(true);
    cv::Mat  res, image;
    cv::Size size        = cv::Size{640, 640};
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.1f;

    std::vector<Object> objs;
     std::string folderPath = "/home/pose/images/";
    std::string fileExtension = "*.*";
    std::vector<cv::String> fileNames;
    cv::glob(folderPath + "*" + fileExtension, fileNames);
	int num = 0;
	for (const auto &filename : fileNames)
    {
        image = cv::imread(filename);
        objs.clear();
        auto start = std::chrono::system_clock::now();
        yolov8_pose->copy_from_Mat(image, size);
        yolov8_pose->infer();
        yolov8_pose->postprocess(objs,image, score_thres, iou_thres, topk);
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        std::cout << " time :" << tc << " ms"<< std::endl;


        yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
        num +=1;
        cv::imwrite("/home/pose/res/"+std::to_string(num)+".jpg",res);
    }
    
    delete yolov8_pose;
    return 0;
}