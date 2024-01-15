# 环境配置与模型下载
## 1. 新建一个虚拟环境

```
conda create --name langchain_demo --clone=/root/share/conda_envs/internlm-base
```

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/7af64357-6777-4294-b4be-713652761d80)

## 2. 进入环境，安装相关依赖
```
conda activate langchain_demo
   # 升级pip
python -m pip install --upgrade pip

pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```
![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/8e8063a0-58bf-4c67-a296-4dc2894c250e)

## 3. langchian 相关环境配置
```
pip install langchain==0.0.292
pip install gradio==4.4.0
pip install chromadb==0.4.15
pip install sentence-transformers==2.2.2
pip install unstructured==0.10.30
pip install markdown==3.3.7
```

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/5e8c143f-e6c5-4a1d-b50d-0d18f3a2b2a5)


## 4. 模型下载
1. LLM 下载

在本地的 /root/share/temp/model_repos/internlm-chat-7b 目录下已存储有所需的模型文件参数，可以直接拷贝到个人目录的模型保存地址：
```
mkdir -p /root/data/model/Shanghai_AI_Laboratory
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/data/model/Shanghai_AI_Laboratory/internlm-chat-7b
```


2. sentence Transformer 下载
我们需要使用到开源词向量模型 Sentence Transformer:（我们也可以选用别的开源词向量模型来进行 Embedding，目前选用这个模型是相对轻量、支持中文且效果较好的，同学们可以自由尝试别的开源词向量模型）

首先需要使用 huggingface 官方提供的 huggingface-cli 命令行工具。安装依赖:
```
pip install -U huggingface_hub
```
然后在 /root/data 目录下新建python文件 download_hf.py，填入以下代码：

resume-download：断点续下
local-dir：本地存储路径。（linux环境下需要填写绝对路径）

```
import os

 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/data/model/sentence-transformer')
```

然后，在 /root/data 目录下执行该脚本即可自动开始下载：

```
python download_hf.py
```

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/a0eb000c-0835-4961-838a-fb4f494f10fe)

## 5. 下载 NLTK 相关资源
我们在使用开源词向量模型构建开源词向量的时候，需要用到第三方库 nltk 的一些资源。正常情况下，其会自动从互联网上下载，但可能由于网络原因会导致下载中断，此处我们可以从国内仓库镜像地址下载相关资源，保存到服务器上。

我们用以下命令下载 nltk 资源并解压到服务器上：

```
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/feed6fa5-e3c8-4807-9b72-51dd87f42a85)

## 6.下载本项目代码
```
cd /root/data
git clone https://github.com/InternLM/tutorial
```
![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/150ac0fe-b595-458a-9e30-fd9d207914dc)


# 知识库搭建
## 1.  数据收集

我们选择由上海人工智能实验室开源的一系列大模型工具开源仓库作为语料库来源，包括：

OpenCompass：面向大模型评测的一站式平台
IMDeploy：涵盖了 LLM 任务的全套轻量化、部署和服务解决方案的高效推理工具箱
XTuner：轻量级微调大语言模型的工具库
InternLM-XComposer：浦语·灵笔，基于书生·浦语大语言模型研发的视觉-语言大模型
Lagent：一个轻量级、开源的基于大语言模型的智能体（agent）框架
InternLM：一个开源的轻量级训练框架，旨在支持大模型训练而无需大量的依赖

```
# 进入到数据库盘
cd /root/data
# clone 上述开源仓库
git clone https://gitee.com/open-compass/opencompass.git
git clone https://gitee.com/InternLM/lmdeploy.git
git clone https://gitee.com/InternLM/xtuner.git
git clone https://gitee.com/InternLM/InternLM-XComposer.git
git clone https://gitee.com/InternLM/lagent.git
git clone https://gitee.com/InternLM/InternLM.git
```
![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/250406c3-6930-4a56-a9a1-de3aa945e211)


