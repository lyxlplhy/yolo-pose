#ifndef POSE_NORMAL_YOLOv8_pose_HPP
#define POSE_NORMAL_YOLOv8_pose_HPP

// 引入TensorRT的插件和一些通用的头文件
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>

using namespace pose;  // 假设pose命名空间包含一些特定的功能或数据结构

// YOLOv8_pose类用于封装YOLOv8姿态估计模型的推理过程
class YOLOv8_pose {
public:
    // 构造函数，传入TensorRT引擎文件路径，用于初始化模型
    explicit YOLOv8_pose(const std::string& engine_file_path,int num_pose,int num_class);

    // 析构函数，用于释放资源
    ~YOLOv8_pose();

    // 创建管道，包括模型加载和预处理配置
    void make_pipe(bool warmup = true);

    // 将OpenCV Mat图像复制到TensorRT的输入内存
    void copy_from_Mat(const cv::Mat& image);

    // 将OpenCV Mat图像调整大小并复制到TensorRT的输入内存
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    // 对图像进行letterbox调整，保证图像长宽比不变
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    // 执行推理
    void infer();

    // 后处理步骤，包括根据置信度和IOU阈值过滤检测结果
    void postprocess(std::vector<Object>& objs, cv::Mat image, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100);

    // 绘制物体检测结果（例如：边界框，关键点）
    void draw_objects(const cv::Mat& image,
                             cv::Mat& res,
                             const std::vector<Object>& objs,
                             const std::vector<std::vector<unsigned int>>& SKELETON,
                             const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                             const std::vector<std::vector<unsigned int>>& LIMB_COLORS);

    // 成员变量
    int num_bindings;      // 输入和输出绑定的总数
    int num_inputs  = 0;   // 输入绑定的数量
    int num_outputs = 0;   // 输出绑定的数量
    std::vector<Binding> input_bindings;   // 输入绑定的信息
    std::vector<Binding> output_bindings;  // 输出绑定的信息
    std::vector<void*> host_ptrs;   // 存储主机内存指针
    std::vector<void*> device_ptrs; // 存储设备内存指针

    PreParam pparam;    // 预处理的参数设置

private:
    // 成员变量
    nvinfer1::ICudaEngine* engine  = nullptr;  // TensorRT引擎
    nvinfer1::IRuntime* runtime = nullptr;     // TensorRT运行时
    nvinfer1::IExecutionContext* context = nullptr; // 执行上下文
    cudaStream_t stream  = nullptr; // CUDA流
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR}; // TensorRT日志记录器，设置为错误级别
    int num_pose=17;//关键点个数
    int num_class=80;//分类别个数
    std::vector<std::string> class_name={"blue","red","black"};
};

#endif  // POSE_NORMAL_YOLOv8_pose_HPP
