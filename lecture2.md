学习链接：https://github.com/InternLM/tutorial/blob/main/helloworld/hello_world.md

## 安装项目实验环境

bash /root/share/install_conda_env_internlm_base.sh xjt_demo

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/1889719a-cb63-407c-a5d3-978983dfacd6)



## 进入环境，并在环境中安装运行 demo 所需要的依赖
```
conda activate xjt_demo
升级pip
python -m pip install --upgrade pip

pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```
<img width="695" alt="101b58f367147e60b862ebdc690bd6a" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/6b59207e-a0f6-4c99-8a40-0880f57bb91f">


## 下载模型
<img width="692" alt="036336eb09f1cdbe8e5b0a613143a3b" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/63c0a681-a915-42e6-ae17-931526aa5413">




# colne 代码
```
git clone https://gitee.com/internlm/InternLM.git
```
<img width="409" alt="ded288f99e9a53e1cbae2c1f922fb0f" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/920d87aa-852d-42c7-9ed2-4cf9187841a9">



# 第一个智能对话demo
![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/6e00a15f-74cc-4e72-b380-b6f225377e83)

## web demo 运行

1. 切换到 VScode 中，运行 /root/code/InternLM 目录下的 web_demo.py 文件

2. 输入以下命令后，将端口映射到本地。
   ```
   cd /root/code/InternLM
   streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
   ```

<img width="947" alt="492ebfe58d39f46a629c5ae4d5ef65c" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/124cb96c-e865-44b9-b89c-f6a7d623c695">

3. 在本地浏览器输入 http://127.0.0.1:6006 即可。

   <img width="960" alt="61f10c25a0afd0ea43c8ef693bd5981" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/2ba84915-3eeb-4c90-99d8-9c908c56676b">

# 第二个 Lagent 智能体工具调用 Demo
有报错

<img width="541" alt="ee50c730306b13b881c0e9c0e0dfe80" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/ed93df04-f9ec-498d-a4bf-229db03300df">



## 浦语·灵笔图文理解创作 Demo

环境配置

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/5f034a31-9763-4d06-ae8b-baf9f843eae5)

运行demo

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/3ad9d539-3e11-4852-aa4c-0255e61ac6c7)
在网页打开demo，体验图文创作

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/f84f3ed7-4a78-4e16-9971-7d9f6f85b4dc)
![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/f8002900-dd46-47d6-9bb6-b27587f03d66)


【这个界面有点不对】











