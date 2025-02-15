#include "NvInferPlugin.h"
#include "fstream"
#include "yolov8-pose.h"


YOLOv8_pose::YOLOv8_pose(const std::string& engine_file_path,int num_pose,int num_class)
{
    this->num_pose=num_pose;
    this->num_class=num_class;
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8_pose::~YOLOv8_pose()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_pose::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8_pose::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::infer()
{
    // 将设备内存中的输入数据传递到推理模型中
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);

    // 对每个输出绑定进行处理，异步将输出从设备内存复制到主机内存
    for (int i = 0; i < this->num_outputs; i++) {
        // 计算每个输出的大小（输出尺寸 * 每个元素的字节大小）
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;

        // 异步将输出数据从设备内存拷贝到主机内存
        // 第一个参数是目标主机内存指针，第二个参数是设备内存指针
        // 使用cudaMemcpyAsync进行异步数据传输，保证高效的并行处理
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }

    // 等待所有的CUDA流操作完成，确保所有数据已正确拷贝到主机内存
    cudaStreamSynchronize(this->stream);
}




void YOLOv8_pose::postprocess(std::vector<Object>& objs, cv::Mat image, float score_thres, float iou_thres, int topk)
{
    // 清空传入的对象列表
    objs.clear();

    // 获取输出绑定的相关信息
    auto& output_binding = output_bindings[0];
    auto num_channels = this->output_bindings[0].dims.d[1]; // 通道数
    auto num_anchors  = this->output_bindings[0].dims.d[2];  // 锚点数
    auto test  = this->output_bindings[0].dims.nbDims;  // 输出维度数
    auto& dw     = this->pparam.dw;  // 输入图像宽度的缩放系数
    auto& dh     = this->pparam.dh;  // 输入图像高度的缩放系数
    auto& width  = this->pparam.width;  // 输入图像宽度
    auto& height = this->pparam.height; // 输入图像高度
    auto& ratio  = this->pparam.ratio;  // 输入图像和输出图像的比例

    // 定义存储边界框、得分、标签、NMS索引和关键点的变量
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    // 将输出绑定数据转换为 OpenCV 的 Mat 格式
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output = output.t();  // 转置矩阵
    int num_labels=3;
    // 遍历每个锚点进行处理
    for (int i = 0; i < num_anchors; i++) {
        // 获取当前锚点的行指针
        auto row_ptr = output.row(i).ptr<float>();
        //std::cout<<"置信度"<<"第一个："<<*(row_ptr+4)<<"第二个："<<*(row_ptr+5)<<"第三个："<<*(row_ptr+6)<<std::endl;
        
        auto flags = output.flags;  // 获取矩阵的标志
        auto bboxes_ptr = row_ptr;  // 边界框数据指针
        auto scores_ptr = row_ptr + 4;  // 得分数据指针
        //std::cout<<"得分"<<(*scores_ptr,*(scores_ptr+1),*(scores_ptr+2))<<std::endl;
        auto kps_ptr = row_ptr + 4 +this->num_class;  // 关键点数据指针

        auto max_s_ptr  = std::max_element(scores_ptr, scores_ptr + this->num_class);  // 获取最大得分的类别指针
        // std::cout<<"最大类别"<<*max_s_ptr<<std::endl;
        // std::cout<<" "<<std::endl;
        float dectect_score = *max_s_ptr;  // 取最大得分作为当前锚点的得分
        // 获取当前锚点的置信度得分
        //float score = *scores_ptr;
        if (dectect_score > score_thres) {  // 如果得分大于阈值，则处理该锚点
            // 解码边界框坐标和尺寸
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            // 将边界框映射到输入图像的坐标范围
            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            // 创建一个 cv::Rect 对象存储边界框
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;
            int label = max_s_ptr - scores_ptr;  // 计算标签（类别编号）
            // 提取关键点数据
            std::vector<float> kps;
            for (int k = 0; k < this->num_pose; k++) {//多少个点就是多少
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);  // 关键点的置信度
                kps_x = clamp(kps_x, 0.f, width);
                kps_y = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);  // 关键点的 x 坐标
                kps.push_back(kps_y);  // 关键点的 y 坐标
                kps.push_back(kps_s);  // 关键点的置信度
            }

            // 将结果添加到存储的向量中
            bboxes.push_back(bbox);

            labels.push_back(label);  // 假设检测为人体目标（标签为 0）
            //std::cout<<"标签有"<<label;
            scores.push_back(dectect_score);
            kpss.push_back(kps);
        }
    }

    // 进行 NMS（非极大值抑制），去除重复的边界框
#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    // 根据 NMS 结果，选取得分最高的 topk 个目标
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {  // 如果达到 topk 上限，则停止
            break;
        }
        Object obj;
        obj.rect = bboxes[i];  // 边界框
        obj.prob = scores[i];  // 置信度
        obj.label = labels[i];  // 标签
        obj.kps = kpss[i];  // 关键点
        objs.push_back(obj);  // 将目标对象添加到输出列表中
        cnt += 1;
    }
}


void YOLOv8_pose::draw_objects(const cv::Mat&                                image,
                               cv::Mat&                                      res,
                               const std::vector<Object>&                    objs,
                               const std::vector<std::vector<unsigned int>>& SKELETON,
                               const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                               const std::vector<std::vector<unsigned int>>& LIMB_COLORS)
{
    res                 = image.clone();
    std::cout << " objs " << objs.size() << std::endl;
    for (auto& obj : objs) {
        int label_index=obj.label;
        std::string label_name=this->class_name[label_index];
        std::cout<<label_name;
        cv::rectangle(res, obj.rect, {0, 0, 255}, 2);

        char text[256];
        sprintf(text, label_name.c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
        // cv::imwrite("/home/xc/pose/res/1.jpg",res);

        auto& kps = obj.kps;
        for (int k = 0; k < this->num_pose*3; k++) {
            if (k < this->num_pose) {
                int   kps_x = std::round(kps[k * 3]);
                int   kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            // auto& ske    = SKELETON[k];
            // int   pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            // int   pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            // int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            // int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            // float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            // float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            // if (pos1_s > 0.5f && pos2_s > 0.5f) {
            //     cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
            //     cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            // }
        }
    }
}