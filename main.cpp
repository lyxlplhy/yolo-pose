#include "opencv2/opencv.hpp"           // 引入OpenCV库，用于图像处理
#include "yolov8-pose.h"                // 引入YOLOv8姿态检测模型
#include "chrono"                        // 引入时间库，用于计算推理时间

// 定义关键点的颜色（绿色）
const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0}};

// 定义人体骨架连接关系，只有一个连接(16, 14)，即连接肩部和肘部
const std::vector<std::vector<unsigned int>> SKELETON = {};

// 定义骨骼颜色（蓝色）
const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255}};

// 主函数
int main(int argc, char** argv)
{
    // 设置CUDA设备为0，确保使用GPU进行加速
    cudaSetDevice(0);

    // 加载YOLOv8姿态检测模型
    auto yolov8_pose = new YOLOv8_pose("/home/pose/model/yolov8s-pose.engine",1,3);//一个点 三个分类
    
    // 初始化YOLOv8姿态检测模型的推理管道
    yolov8_pose->make_pipe(true);

    // 声明变量
    cv::Mat res, image;
    cv::Size size = cv::Size{640, 640}; // 设置输入图像的尺寸
    int topk = 100; // 设置最多检测到的物体数量
    float score_thres = 0.25f; // 设置得分阈值，低于该阈值的物体将被丢弃
    float iou_thres = 0.1f; // 设置IOU阈值，用于非极大值抑制

    std::vector<Object> objs; // 用于存储检测到的物体
    std::string folderPath = "/home/pose/images/"; // 图像文件夹路径
    std::string fileExtension = "*.*"; // 文件扩展名通配符，表示所有文件
    std::vector<cv::String> fileNames; // 存储所有图像文件名

    // 获取文件夹中的所有图像文件
    cv::glob(folderPath + "*" + fileExtension, fileNames);

    int num = 0; // 记录处理的图像编号
    // 遍历所有图像文件
    for (const auto &filename : fileNames)
    {
        // 读取图像
        image = cv::imread(filename);

        // 清空物体列表
        objs.clear();

        // 记录开始时间
        auto start = std::chrono::system_clock::now();

        // 将图像复制到YOLOv8模型并进行推理前处理
        yolov8_pose->copy_from_Mat(image, size);

        // 执行推理
        yolov8_pose->infer();

        // 处理推理结果，提取物体，并进行后处理
        yolov8_pose->postprocess(objs, image, score_thres, iou_thres, topk);

        // 记录结束时间
        auto end = std::chrono::system_clock::now();

        // 计算推理时间（毫秒）
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        std::cout << " time :" << tc << " ms" << std::endl;

        // 绘制检测结果（关键点和骨架）
        yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);

        // 更新图像编号
        num += 1;

        // 保存处理后的图像
        cv::imwrite("/home/pose/res/" + std::to_string(num) + ".jpg", res);
    }

    // 释放YOLOv8姿态检测模型的内存
    delete yolov8_pose;

    return 0;
}
