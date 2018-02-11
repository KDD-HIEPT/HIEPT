# HIEPT
source code for HIEPT

## Requirements
-  tensorflow==1.3.0
-  numpy==1.14.0
-  bitarray==0.8.1
-  matplotlib==1.5.11

## DataSets
We crawled four datasets to evaluate HIEPT including twitter, weibo, zhihu, douban. Due to privacy protection, we do not provide the raw data. Fortunately, we provide the embedding data for both tweets and social networks, which you can find in the ` data`  folder.

In ` data` folder, we can see four datasets. There are three subfolders in each dataset:
- graph_embedding:embedding for social networks, include [DeepWalk](https://github.com/phanein/deepwalk), [LINE](https://github.com/tangjianpku/LINE), [node2vec](https://github.com/aditya-grover/node2vec), [CANE](https://github.com/thunlp/cane), [TADW](https://github.com/thunlp/TADW),[struc2vec](https://github.com/leoribeiro/struc2vec)
- topics:user joined topics, each line means a topic,each number in a line means a userid
- tweets_embedding:embedding for user's tweets, we use [gensim](https://radimrehurek.com/gensim/models/doc2vec.html) to implement [doc2vec](https://arxiv.org/pdf/1405.4053v2.pdf)

## Usage
To run HIEPT, use the following code in terminal at `src` folder:
> python main.py

Or clone the project to you python IDE(eg:Pycharm)
## Configuration
Change the configuration in `src/config.py`:
- data_set:which dataset you want to run,available datasets:`douban`, `weiobo`, `twitter`, `zhihu`
- ne_type:the network embedding type,available parameters:`node2vec`, `line`, `tadw`, `deepwalk`, `stuct2vec`, `cane`

You can find more configuration params in `src/config.py`