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


## 四、快速使用

MindFormers套件对外提供两种使用和开发形式，为开发者提供灵活且简洁的使用方式和高阶开发接口。

### 方式一：使用已有脚本启动

用户可以直接clone整个仓库，按照以下步骤即可运行套件中已支持的任意`configs`模型任务配置文件，方便用户快速进行使用和开发：

- 准备工作

    - step1：git clone mindformers

  ```shell
  git clone -b r1.0 https://gitee.com/mindspore/mindformers.git
  cd mindformers
  ```

    - step2:  准备相应任务的数据集，请参考`docs`目录下各模型的README.md文档准备相应数据集

    - step3：修改配置文件`configs/{model_name}/run_{model_name}_***.yaml`中数据集路径

    - step4：如果要使用分布式训练，则需提前生成RANK_TABLE_FILE

  ```shell
  # 不包含8本身，生成0~7卡的hccl json文件。注意：不支持在镜像容器中执行该命令，请在容器外执行。
  python mindformers/tools/hccl_tools.py --device_num [0,8)
  ```

- 单卡启动：统一接口启动，根据模型 CONFIG 完成任意模型的单卡训练、微调、评估、推理流程

```shell
# 训练启动，run_mode支持train、finetune、eval、predict四个关键字，以分别完成模型训练、评估、推理功能，默认使用配置文件中的run_mode
python run_mindformer.py --config {CONFIG_PATH} --run_mode {train/finetune/eval/predict}
```

- 多卡启动： scripts 脚本启动，根据模型 CONFIG 完成任意模型的单卡/多卡训练、微调、评估、推理流程

    - 使用 [rank table方式启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/rank_table.html)

      ```shell
      # 8卡分布式运行， DEVICE_RANGE = [0,8), 不包含8本身
      cd scripts
      bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_MODE
      ```

    - 使用[动态组网方式启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/dynamic_cluster.html)

      ```shell
      # 8卡分布式运行
      启动前的准备:
      1. 使用hostname命令将每台服务器hostname设置为各自的ip:  hostname [host ip], 如果在docker内需求设置为docker内部ip,同时保证各个服务器之间docker网络互通
      2. 设置环境变量: export SERVER_ID=0; export SERVER_NUM=1; export PER_DEVICE_NUMS=8; export MS_SCHED_HOST=[HOST IP]; export MS_SCHED_PORT=[PORT]
      cd scripts
      # SERVER_ID为当前服务器序号，SERVER_NUM为服务器的总数，PER_DEVICE_NUMS为每台服务器使用的卡数默认值为8，MS_SCHED_HOST为调度节点的ip，MS_SCHED_PORT为通信端口
      bash run_distribute_ps_auto.sh CONFIG_PATH RUN_MODE
      ```

- 常用参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的{model_name}/run_*.yaml配置文件
DEVICE_ID: 为设备卡，范围为0~7
DEVICE_RANGE: 为单机分布式卡的范围, 如[0,8]为8卡分布式，不包含8本身
RUN_MODE: 为任务运行状态，支持关键字 train\finetune\eval\predict\export
```

### 方式二：调用API启动

**详细高阶API使用教程请参考：**[MindFormers大模型使用教程](docs/readthedocs/source_zh_cn/docs/practice/Develop_With_Api.md)

- 准备工作

    - step 1：安装mindformers

  具体安装请参考[第二章](https://gitee.com/mindspore/mindformers/blob/dev/README.md#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)。

    - step2: 准备数据

  准备相应任务的数据集，请参考`docs`目录下各模型的README.md文档准备相应数据集。

- Trainer 快速入门

  用户可以通过以上方式安装mindformers库，然后利用Trainer高阶接口执行模型任务的训练、微调、评估、推理功能。

    - Trainer 训练/微调启动

  用户可使用`Trainer.train`或者`Trainer.finetune`接口完成模型的训练/微调/断点续训。

  ```python
  import mindspore; mindspore.set_context(mode=0, device_id=0)
  from mindformers import Trainer

  cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                        model='vit_base_p16', # 已支持的模型名
                        train_dataset="/data/imageNet-1k/train", # 传入标准的训练数据集路径，默认支持ImageNet数据集格式
                        eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式
  # Example 1： 开启训练复现流程
  cls_trainer.train()
  # Example 2： 加载集成的mae权重，开启微调流程
  cls_trainer.finetune(finetune_checkpoint='mae_vit_base_p16')
  # Example 3： 开启断点续训功能
  cls_trainer.train(train_checkpoint=True, resume_training=True)
  ```

    - Trainer 评估启动

  用户可使用`Trainer.evaluate`接口完成模型的评估流程。

  ```python
  import mindspore; mindspore.set_context(mode=0, device_id=0)
  from mindformers import Trainer

  cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                        model='vit_base_p16', # 已支持的模型名
                        eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式
  # Example 1： 开启评估已集成模型权重的复现流程
  cls_trainer.evaluate()
  # Example 2： 开启评估训练得到的最后一个权重
  cls_trainer.evaluate(eval_checkpoint=True)
  # Example 3： 开启评估指定的模型权重
  cls_trainer.evaluate(eval_checkpoint='./output/checkpoint/rank_0/mindformers.ckpt')
  ```

  ```text
  结果打印示例(已集成的vit_base_p16模型权重评估分数)：
  Top1 Accuracy=0.8317
  ```

    - Trainer 推理启动

  用户可使用`Trainer.predict`接口完成模型的推理流程。

  ```python
  import mindspore; mindspore.set_context(mode=0, device_id=0)
  from mindformers import Trainer

  cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                        model='vit_base_p16') # 已支持的模型名
  input_data = './cat.png' # 一张猫的图片
  # Example 1： 指定输入的数据完成模型推理
  predict_result_d = cls_trainer.predict(input_data=input_data)
  # Example 2： 开启推理（自动加载训练得到的最后一个权重）
  predict_result_b = cls_trainer.predict(input_data=input_data, predict_checkpoint=True)
  # Example 3： 加载指定的权重以完成推理
  predict_result_c = cls_trainer.predict(input_data=input_data, predict_checkpoint='./output/checkpoint/rank_0/mindformers.ckpt')
  print(predict_result_d)
  ```

  ```text
  结果打印示例(已集成的vit_base_p16模型权重推理结果)：
  {‘label’: 'cat', score: 0.99}
  ```

- pipeline 快速入门

  MindFormers套件为用户提供了已集成模型的pipeline推理接口，方便用户体验大模型推理服务。

    - pipeline 使用

  ```python
  # 以gpt2 small为例
  import mindspore; mindspore.set_context(mode=0, device_id=0)
  from mindformers.pipeline import pipeline

  pipeline_task = pipeline(task="text_generation", model="gpt2")
  pipeline_result = pipeline_task("An increasing sequence: one,", do_sample=False, max_length=20)
  print(pipeline_result)
  ```

  ```text
  结果打印示例(已集成的gpt2模型权重推理结果)：
  [{'text_generation_text': ['An increasing sequence: one, two, three, four, five, six, seven, eight,']}]
  ```

- AutoClass 快速入门

  MindFormers套件为用户提供了高阶AutoClass类，包含AutoConfig、AutoModel、AutoProcessor、AutoTokenizer四类，方便开发者进行调用。

    - AutoConfig 获取已支持的任意模型配置

  ```python
  from mindformers import AutoConfig

  # 获取gpt2的模型配置
  gpt2_config = AutoConfig.from_pretrained('gpt2')
  # 获取vit_base_p16的模型配置
  vit_base_p16_config = AutoConfig.from_pretrained('vit_base_p16')
  ```

    - AutoModel 获取已支持的网络模型

  ```python
  from mindformers import AutoModel

  # 利用from_pretrained功能实现模型的实例化（默认加载对应权重）
  gpt2 = AutoModel.from_pretrained('gpt2')
  # 利用from_config功能实现模型的实例化（默认加载对应权重）
  gpt2_config = AutoConfig.from_pretrained('gpt2')
  gpt2 = AutoModel.from_config(gpt2_config)
  # 利用save_pretrained功能保存模型对应配置
  gpt2.save_pretrained('./gpt2', save_name='gpt2')
  ```

    - AutoProcessor 获取已支持的预处理方法

  ```python
  from mindformers import AutoProcessor

  # 通过模型名关键字获取对应模型预处理过程（实例化gpt2的预处理过程，通常用于Trainer/pipeline推理入参）
  gpt2_processor_a = AutoProcessor.from_pretrained('gpt2')
  # 通过yaml文件获取相应的预处理过程
  gpt2_processor_b = AutoProcessor.from_pretrained('configs/gpt2/run_gpt2.yaml')
  ```

    - AutoTokenizer 获取已支持的tokenizer方法

  ```python
  from mindformers import AutoTokenizer
  # 通过模型名关键字获取对应模型预处理过程（实例化gpt2的tokenizer，通常用于Trainer/pipeline推理入参）
  gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
  ```

## 五、贡献

欢迎参与社区贡献，可参考MindSpore贡献要求[Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING_CN.md)。

## 六、许可证

[Apache 2.0许可证](LICENSE)
