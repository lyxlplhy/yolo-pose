# 大模型关键参数识别mindspore实现

## 一、使用华为modelart平台

### 设置启动方式和代码目录

启动方式中选择预置框架，Ascend-Powered-Engine mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b，代码目录选择obs目录中代码目录，obs使用指南参考[obs指南](https://support.huaweicloud.com/intl/zh-cn/browsertg-obs/obs-browsertg-zh-pdf.pdf)，启动文件选择 /mindformers-r1.0/run_mindformer.py
![image](https://github.com/user-attachments/assets/d8131ceb-eade-4461-bb61-329b01abb791)

### 超参数设置
config为配置文件路径，remote_save_url为输出obs路径，具体路径由放置环境而定

config：configs/llama2/run_llama2_7b.yaml

use_parallel： True

run_mode： finetune

remote_save_url： obs://xinanllama/out/
![image](https://github.com/user-attachments/assets/97df3c61-6a3c-4457-b031-6ccb60ea23a3)

### 环境变量设置
GE_NOT_CUT=1

MA_DETECT_TRAIN_INJECT_CODE=0

MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE

MS_DEV_SIDE_EFFECT_LOAD_ELIM=3

MS_MEMORY_POOL_RECYCLE=1

![image](https://github.com/user-attachments/assets/ecf687fa-22d2-4b17-baa3-ed5eda851e7d)

### 推荐硬件
实例规格：Ascend: 8*Ascend-snt9b1(32GB)|arm:192核 1536GB

![image](https://github.com/user-attachments/assets/42ed0635-4953-4b7f-afd3-afeb098f9f34)





