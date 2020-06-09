import numpy as np
import pandas as pd
import re,os,jieba
from itertools import chain
"""第一步：把文档划分成句子"""

# 文档所在的文件夹
c_root = os.getcwd()+os.sep+"cnews"+os.sep

sentences_list = []
for file in os.listdir(c_root):
    fp = open(c_root+file,'r',encoding="utf8")
    for line in fp.readlines():
        if line.strip():
            # 把元素按照[。！；？]进行分隔，得到句子。
            line_split = re.split(r'[。！；？]',line.strip())
            # [。！；？]这些符号也会划分出来，把它们去掉。
            line_split = [line.strip() for line in line_split if line.strip() not in ['。','！','？','；'] and len(line.strip())>1]
            sentences_list.append(line_split)
sentences_list = list(chain.from_iterable(sentences_list))
print("前10个句子为：\n")
print(sentences_list[:10])

"""第二步：文本预处理，去除停用词和非汉字字符,并进行分词"""


#创建停用词列表
stopwords = [line.strip() for line in open('./stopwords.txt',encoding='UTF-8').readlines()]

# 对句子进行分词
def seg_depart(sentence):
    # 去掉非汉字字符
    sentence = re.sub(r'[^\u4e00-\u9fa5]+','',sentence)
    sentence_depart = jieba.cut(sentence.strip())
    word_list = []
    for word in sentence_depart:
        if word not in stopwords:
            word_list.append(word)
    # 如果句子整个被过滤掉了，如：'02-2717:56'被过滤，那就返回[],保持句子的数量不变
    return word_list

sentence_word_list = []
for sentence in sentences_list:
    line_seg = seg_depart(sentence)
    sentence_word_list.append(line_seg)
print("一共有",len(sentences_list),'个句子。\n')
print("前10个句子分词后的结果为：\n",sentence_word_list[:10])

# 保证处理后句子的数量不变，我们后面才好根据textrank值取出未处理之前的句子作为摘要。
if len(sentences_list) == len(sentence_word_list):
    print("\n数据预处理后句子的数量不变！")

"""第三步：准备词向量"""

word_embeddings = {}
f = open('./sgns.financial.char', encoding='utf-8')
for line in f:
    # 把第一行的内容去掉
    if '467389 300\n' not in line:
        values = line.split()
        # 第一个元素是词语
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = embedding
f.close()
print("一共有"+str(len(word_embeddings))+"个词语/字。")

"""第四步：得到词语的embedding，用WordAVG作为句子的向量表示"""

sentence_vectors = []
for i in sentence_word_list:
    if len(i)!=0:
        # 如果句子中的词语不在字典中，那就把embedding设为300维元素为0的向量。
        # 得到句子中全部词的词向量后，求平均值，得到句子的向量表示
        v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i])/(len(i))
    else:
        # 如果句子为[]，那么就向量表示为300维元素为0个向量。
        v = np.zeros((300,))
    sentence_vectors.append(v)

"""第四步：计算句子之间的余弦相似度，构成相似度矩阵"""
sim_mat = np.zeros([len(sentences_list), len(sentences_list)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences_list)):
  for j in range(len(sentences_list)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
print("句子相似度矩阵的形状为：",sim_mat.shape)

"""第五步：迭代得到句子的textrank值，排序并取出摘要"""
import networkx as nx

# 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
nx_graph = nx.from_numpy_array(sim_mat)

# 得到所有句子的textrank值
scores = nx.pagerank(nx_graph)

# 根据textrank值对未处理的句子进行排序
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences_list)), reverse=True)

# 取出得分最高的前10个句子作为摘要
sn = 10
for i in range(sn):
    print("第"+str(i+1)+"条摘要：\n\n",ranked_sentences[i][1],'\n')