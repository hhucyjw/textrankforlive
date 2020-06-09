### 基于**TextRank的**直播带货行业热点感知

#### 摘要

文章介绍了一种使用TextRank算法做直播带货领域多文本的自动摘要方法，可以实现直播带货行业热点感知，在短时间内获取直播带货行业当日热点中的热点。具体实现步骤为：使用爬虫工具爬取当日各大媒体的直播带货相关最新文章；把所有文章整合成文本数据，并把文本分割成单个句子；用WordAVG的方法，将每个句子中所有单词的词向量合并为句子的向量表示；计算句子向量间的相似性并存放在矩阵中，作为转移概率矩阵M；然后将转移概率矩阵转换为以句子为节点、相似性得分为边的图结构，用于句子TextRank计算；对句子按照TextRank值进行排序，排名最靠前的n个句子作为摘要。文章使用了来自Yuanyuan Qiu, Hongzheng Li, Shen Li, Yingdi Jiang, Renfen Hu, Lijiao Yang. [*Revisiting Correlations between Intrinsic and Extrinsic Evaluations of Word Embeddings*](http://www.cips-cl.org/static/anthology/CCL-2018/CCL-18-086.pdf). Chinese Computational Linguistics and Natural Language Processing Based on Naturally Annotated Big Data. Springer, Cham, 2018. 209-221. (CCL & NLP-NABD 2018 Best Paper)的，预处理完成的，开源金融新闻领域word2vec词向量，其训练方法为Skip-gram的负采样方法。基于以上方法，可以实现直播带货行业热点感知，解释为：TextRank算法对于文章中的每一句话都会给出一个正实数值，表示这段话的重要程度，TextRank越高，表示这段话越重要。在直播带货行业日趋激烈的大环境下，一天产生的新闻量也是巨大的，该方法可以在短时间内提取所有新闻中的热点，并按照其重要程度进行排序，以实现在大量的直播带货新闻中以最快速度感知当日热点新闻中的热点，及时了解行业走向并辅助决策实施。

#### 附录

1.“cnews”文件夹下为2020.6.9当日的直播带货有关新闻

2.“stopwords.txt”为文本处理中使用的停用词

3.“直播带货行业热点感知.py”为项目实现的python代码

4.“sgns.financial.char”为预处理word2vec词向量文件（https://github.com/Embedding/Chinese-Word-Vectors）



