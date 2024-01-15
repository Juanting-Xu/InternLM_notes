# 大模型局限性：
1. 知识时效性受限：如何让LLM能够获得最新的知识
2. 专业能力有限：如何打造垂域大模型
3. 定制化成本高：如何打造个人专属的LLM应用

# 大模型开发范式：
## RAG：检索增强生成
核心思想：给大模型外挂一个知识库，对于用户的提问，会首先从知识库中匹配到相关文档，然后将文档和提问一起交给大模型来生成回答，从而提高大模型的知识储备

<img width="271" alt="a489b12e8e3b445ed63296584eccf7e" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/3ff83084-d415-4023-88c7-d9daf08dd86e">

优缺点：
1. 成本低，无需GPU算力、可实时更新
2. 其能力受基座模型的影响大，基座模型能力的上限极大决定了RAG应用能力的天花板；
3. query和document一起输入到大模型来回答，占用了大量的模型上下文，因此单次回答知识有限；因此 对于一些需要大跨度收集知识进行总结性回答的问题表现不佳



## Finetune

核心思想：在较小的数据集上进行轻量级的训练微调，从而提升模型在这个新数据集上的能力
优缺点：
1. 可个性化微调
2. 知识覆盖面广
3. 成本高昂，需要很多GPU算力和数据
4. 无法实时更新

# Langchain
![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/1e159ab7-67c7-4c75-aa17-968c86ba2fe5)

## 基于LangChain搭建RAG应用

<img width="399" alt="9836b879edf6490e0e0eb54906d108a" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/cc9cc28b-6e11-4aed-a923-c47b02b3e2e3">
1. 加载数据：使用Unstructed Loader来加载本地文档，并转成统一的纯文本格式
2. 文本分割：将文档划分成Chunk
3. 向量化：使用sentence Transformer 将文本段转成向量格式，存储到Chroma数据库中
4. 使用sentence Transformer 对用户输入的query进行向量化，得到Query Vector
5. 相似度匹配：Query Vector与向量数据库中的文本段进行相似度匹配，召回相关的文本段
6. 将query 和相关的文本段（Text Chunk）嵌入已经写好的Prompt template中，调用 LLM 进行回答，得到答案

# 构建向量数据库

![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/956b19ab-f0c4-4cbb-827c-a228aa47dab4)

# 基于LangChain构建知识助手
将InternLM接入Langchain

<img width="505" alt="c2067300ab5ef82e53527d229d6fdcd" src="https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/6c9b36ad-655c-4977-b53f-ffe4d5631b77">


![image](https://github.com/Juanting-Xu/InternLM_notes/assets/36044048/71d03c98-58b8-4a75-b7cd-544393221b1f)





