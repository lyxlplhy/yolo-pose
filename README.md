# 大模型关键参数识别mindspore实现


## 一、使用华为modelarts平台

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

### 启动微调

点击提交，输出文件训练日志为设置remote_save_url地址

![image](https://github.com/user-attachments/assets/ee857f33-e641-4462-8bdf-b5c5322415ad)


## 二、配置文件及权重转换

### 基础配置

#### 模型权重
创建文件夹
```
cd mindspore 

mkdir ckpt

cd ckpt
```

在notebook的Terminal命令行输入 python，进入python交互界面，然后输入如下语句：
```
import moxing as mox
mox.file.copy_parallel("obs://hb-public/LLM/llama2/llama2-ckpt.tar.gz", "/home/ma-user/work/mindformers/ckpt/llama2-ckpt.tar.gz")
```

退出交互界面，将文件解压：
```
tar -zxvf llama2-ckpt.tar.gz

注意：如果下载权重为huggingface权重，需要转化为ckpt权重
```
转化为mindspore格式权重：
```
python mindformers/models/llama/convert_weight.py \
--torch_ckpt_path TORCH_CKPT_PATH \
--mindspore_ckpt_path {path}/MS_CKPT_NAME
```
```
# 参数说明
torch_ckpt_path: huggingface权重保存目录下的任意权重bin文件,根据该文件路径读取目录下全部权重
mindspore_ckpt_path: 权重保存文件名，可以指定自定义保存路径

```
#### 训练数据转换

以Wikitext2数据集为例，[下载](https://hb-public.obs.cn-north-300.hblfrgzn.com:443/datasets/wikitext-2-v1.zip)，具体[参考](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87-%E9%A2%84%E8%AE%AD%E7%BB%83)

```
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /home/ma-user/work/mindformers/datasets/wikitext-2/wiki.train.tokens \
--model_file /home/ma-user/work/mindformers/ckpt/llama2/tokenizer.model \
--seq_length 4096 \
--output_file /home/ma-user/work/mindformers/mr_datasets/wiki4096.mindrecord
```

#### 配置文件
data_dir为训练数据和验证数据路径，load_checkpoint为模型权重路径

data_dir：/home/ma-user/work/mindformers/mr_datasets/wiki4096.mindrecord

load_checkpoint: 'mindformers-r1.0/data/llama2_7b.ckpt'

## 三、攻击方式

攻击方式设置，两种模式设置，在llama_mindspore/mindformer/trainer/base_trainer 759行，设置True或者False，实现全模型攻击和针对攻击单层恢复

防御方式设置，设置防御多少层级：llama_mindspore/mindformer/trainer/base_trainer 754行，设置模型被保护的层级的名字，具体名字根据模型情况而定

```
parameters={param.name: param.data for param in model.train_network.network.get_parameters()}
        for name in parameters.keys():
            print(name)
        logger.info("delete")
        for param in model.train_network.network.get_parameters():
            if 'layers.1' in param.name or 'layers.2' in param.name:
                logger.info(param.name)
                logger.info("is initialization")
                param.set_data(initializer(Normal(), param.data.shape, param.data.dtype))
            else:
                param.trainable = False
```





