# 环境配置
1. 新建一个虚拟环境

```
conda create --name langchain_demo --clone=/root/share/conda_envs/internlm-base
```

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/7af64357-6777-4294-b4be-713652761d80)

2. 进入环境，安装相关依赖
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

3. langchian 相关环境配置
```
pip install langchain==0.0.292
pip install gradio==4.4.0
pip install chromadb==0.4.15
pip install sentence-transformers==2.2.2
pip install unstructured==0.10.30
pip install markdown==3.3.7
```

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/5e8c143f-e6c5-4a1d-b50d-0d18f3a2b2a5)


# 模型下载
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

3. 下载 NLTK 相关资源

