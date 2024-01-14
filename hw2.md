# 使用 InternLM-Chat-7B 模型生成 300 字的小故事
![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/c8a670d2-b156-4ece-935d-ce2767ad4c99)


# 使用 huggingface_hub python 包 下载config文件
1. 安装hugging-face
```
pip install -U huggingface_hub
```
2. 使用 huggingface_hub 下载模型中的部分文件,脚本如下：
```
import os
from huggingface_hub import hf_hub_download  # Load model directly 

hf_hub_download(repo_id="internlm/internlm-20b", filename="config.json",
                local_dir="/root/model/internlm-20b/" , endpoint="https://hf-mirror.com")
```
3. 下载完成

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/546918bb-d872-4e72-8fe4-bf19f4b5000d)



# 浦语·灵笔图文理解创作 Demo

## 图文创作

![042d428977eb9afdfbe23752858eed9](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/a63302e8-511f-4863-99f6-92da3a8eed7c)


## 多模态对话

![88f443db1097b76affe93110570ee21b_](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/f2640628-0751-48be-9363-2a0430526e9b)


## Lagent 工具调用 Demo 创作

![2df928249ee3f3909090c789cb915ed](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/a6252076-149c-477d-9d48-63291f8982fe)






