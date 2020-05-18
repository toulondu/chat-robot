from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# 将文件的每一行拆分为字段字典, 读movie_lines.txt，每一行是一句对话，key为lineID
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

# 将 `loadLines` 中的行字段分组为基于 *movie_conversations.txt* 的对话
# 每一行是一次完整对话，包含多个lineID，至此就转换成了一组一组完整的对话
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# 从对话中提取一对句子，相当于每两句话组成一个样本
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

# 接下来我们把数据处理成我们想要的样子存放在一个新的文件中便于读取
# 定义新文件的路径
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

if not os.path.exists(datafile):
    # 用制表符来分隔句子
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # 初始化行dict，对话列表和字段ID
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # 加载行和进程对话
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                    lines, MOVIE_CONVERSATIONS_FIELDS)

    # 写入新的csv文件 todo..改成pandas
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)


# 默认词向量
PAD_token = 0  # 填充短句
SOS_token = 1  # 句子起始标记
EOS_token = 2  # 句子结束标记

# 数据加载和清洗
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除低于特定计数阈值的单词，减少特征空间维度，从而降低学习目标函数的难度，加快收敛
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # 重初始化字典
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

# 进一步清洗数据
MAX_LENGTH = 10  # 最大句子长度，超出舍去

# 将Unicode字符串转换为纯ASCII，多亏了
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    # NFD表示字符应该分解为多个组合字符表示，将类型为音调符号的所有字符删除
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 初始化Voc对象 和 格式化pairs对话存放到list中
def readVocs(datafile):
    print("Reading lines...")
    # 读取所有句子到数组中
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[unicodeToAscii(s) for s in l.split('\t')] for l in lines]
    return pairs

# 如果对 'p' 中的两个句子都低于 MAX_LENGTH 阈值，则返回True，即过长对话不要
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 过滤满足条件的 pairs 对话
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 使用上面定义的函数，返回一个填充的voc对象和对列表
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")

    return pairs

MIN_COUNT = 3    # 修剪的最小字数阈值

def trimRareWords(voc, pairs, MIN_COUNT=MIN_COUNT):
    # 修剪来自voc的MIN_COUNT下使用的单词
    voc.trim(MIN_COUNT)
    
    # 同时，如果句子中包含生僻词，也删除句子
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 检查输入句子
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 检查输出句子
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # 只保留输入或输出句子中不包含修剪单词的对
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

paris_path = './pairs.json'
if not os.path.exists(paris_path):
    # 加载/组装voc和对
    save_dir = os.path.join("data", "save")
    pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

    # 修剪voc和对
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    json.dump(pairs, open(paris_path, 'w'))
else:
    pairs = json.load(open(paris_path))

voc = Voc(corpus_name)
for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
print("Counted words:", voc.num_words)

# 将句子中的词替换为相应word id
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# zip_longest 从每个句子中一个一个拿词组成新的列表，长度以最长句子为准，不足补PAD_token， 输出数据相当于在填充原数据的基础上进行了行列转置
# 转置的目的是为了更方便按照时间序列取出单词，[batch_size,max_lenth]结构用下标0取出的是第一个句子，而[max_length,batch_size]用下标0取出的则是该批次句子的第一个time_step的句子。这对于我们使用时序模型来说非常方便
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# 记录 PAD_token的位置为0， 其他的为1
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# 返回填充前（加入结束index EOS_token做标记）的长度 和 填充后的输入序列张量
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# 返回填充前（加入结束index EOS_token做标记）最长的一个长度 和 填充后的输入序列张量, 和 填充后的标记 mask
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# 返回给定batch对的所有项目
def batch2TrainData(voc, pair_batch):
    """
    return:
        inp: 按batch中最大句子长度进行填充后的输入句id列表
        lengths: batch中输入句子填充前的长度列表
        output：按batch中最大句子长度进行填充后的输出句id列表
        mask: batch中输出句的掩膜，有词部分为1，填充部分为0
        max_target_len: batch中所有输出句 的最大长度
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

# 验证例子
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

# 我们这里选择seq2seq作为生成模型，所以我们接下来对 encoder和decoder进行编码 todo..
