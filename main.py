# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models
import logging
import jieba
import jieba.posseg
import os
from gensim.models.doc2vec import Doc2Vec, LabeledSentence, TaggedDocument
import numpy as np

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F     # 激励函数都在这

#進行中文分詞並打標記，放到x_train供後續索引
def  cut_sentence():

    fp = open('waimai_10k_zh_tw.csv', "r")
    line = fp.readline()        # 第一行是label,review
    line = fp.readline()

    cut_sent_list = []
    train_sent_num = 3000
    test_sent_num = 1000
    train_positive_num = 0
    train_negative_num = 0
    test_positive_num = 0
    test_negative_num = 0
    positive_index = []
    negative_index = []
    positive_train = []
    negative_train = []
    positive_test = []
    negative_test = []

    index = 0

    # 用 while 逐行讀取檔案內容，直至檔案結尾
    while line:
        sent = ''
        sent = line.replace('\n', '')
        sent = sent[2:]
        cut_sent =' '.join(list(jieba.cut(sent)))
        cut_sent_list.append(cut_sent)

        if line[:2] == '1,':
            if train_positive_num < train_sent_num:
                positive_index.append(index)
                positive_train.append(cut_sent)
                train_positive_num += 1
            elif test_positive_num < test_sent_num:
                positive_test.append(cut_sent)
                test_positive_num += 1
        else:
            if train_negative_num < train_sent_num:
                negative_index.append(index)
                negative_train.append(cut_sent)
                train_negative_num += 1
            elif test_negative_num < test_sent_num:
                negative_test.append(cut_sent)
                test_negative_num += 1
        
        line = fp.readline()
        index += 1
    
    fp.close()

    x_train = []
    for i, text in enumerate(cut_sent_list):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        # print(word_list)
        document = TaggedDocument(word_list, tags=[i])
        x_train.append(document)

    model_dm = train(x_train)

    SaveSentenceIndex('positive_index.txt', model_dm, positive_index)
    SaveSentenceIndex('negative_index.txt', model_dm, negative_index)
    SaveSentence('positive_train.txt', positive_train)
    SaveSentence('negative_train.txt', negative_train)
    SaveSentence('positive_test.txt', positive_test)
    SaveSentence('negative_test.txt', negative_test)

# 模型训练
def train(x_train):
    model_dm = Doc2Vec(x_train, vector_size=200, window=5, min_count=1, workers=4, epochs = 10)
    model_dm.save('sent2vec.model')
    
    return model_dm

# 將要訓練的句子位置存起來
def SaveSentenceIndex(filepath, model_dm, index_list):
    f = open(filepath, 'w', encoding='UTF-8')
    for index in index_list:
        f.write(str(index) + '\n')
    f.close()

# 將要訓練的句子(斷詞後)存起來
def SaveSentence(filepath, sent_list):
    f = open(filepath, 'w', encoding='UTF-8')
    for sent in sent_list:
        f.write(sent + '\n')
    f.close()

#实例
def test():
    model_dm = Doc2Vec.load('sent2vec.model')
    # 計算指定的兩個句子的相似度
    # sims = model_dm.docvecs.similarity(18,20)
    # print(sims)
    # 顯示第一個句子的向量
    # 给定文档进行测试，并计算相似度，取前10高的顯示
    sent = '很快，好吃，味道足，量大'
    test_text = ' '.join(list(jieba.cut(sent)))
    # print(test_text)  # 很快 ， 好吃 ， 味道 足 ， 量 大

    # word_list = test_text.split(' ')
    # print(word_list)
    # inferred_vector_dm = model_dm.infer_vector(word_list)         # 將句子透過模型獲得向量
    # # print(inferred_vector_dm)
    # sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    # print(sims)

# 使用神經網路
def useInt():
    # 讀取正負句子的向量
    train_positive_vector = []
    train_negative_vector = []
    test_positive_vector = []
    test_negative_vector = []

    # 使用word2vec向量取句子向量
    model = models.Word2Vec.load('word2vec.model')

    LoadVector('positive_train.txt', model, train_positive_vector)
    LoadVector('negative_train.txt', model, train_negative_vector)
    LoadVector('positive_test.txt', model, test_positive_vector)
    LoadVector('negative_test.txt', model, test_negative_vector)

    # 使用doc2vec取句子向量
    # model_dm = Doc2Vec.load('sent2vec.model')
    # f = open('positive_index.txt','r', encoding='UTF-8')
    # line = f.readline()
    # while line:
    #     index = line.replace('\n','')
    #     positive_vector.append(model_dm.docvecs[int(index)])
    #     line = f.readline()
    # f.close()
    
    # f = open('negative_index.txt','r', encoding='UTF-8')
    # line = f.readline()
    # while line:
    #     index = line.replace('\n','')
    #     negative_vector.append(model_dm.docvecs[int(index)])
    #     line = f.readline()
    # f.close()

    # 訓練数据
    train_positive = torch.FloatTensor(train_positive_vector)
    train_y0 = torch.zeros(3000)               # 类型0 y data (tensor), shape=(100, )
    train_negative = torch.FloatTensor(train_negative_vector)
    train_y1 = torch.ones(3000)                # 类型1 y data (tensor), shape=(100, )

    # 測試数据
    test_positive = torch.FloatTensor(test_positive_vector)
    test_y0 = torch.zeros(1000)               # 类型0 y data (tensor), shape=(100, )
    test_negative = torch.FloatTensor(test_negative_vector)
    test_y1 = torch.ones(1000)                # 类型1 y data (tensor), shape=(100, )

    # 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
    train_input_x = torch.cat((train_negative, train_positive), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
    train_target_y = torch.cat((train_y0, train_y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer target依照0,1,2..
    # 測試的input&target
    test_input_x = torch.cat((test_negative, test_positive), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
    test_target_y = torch.cat((test_y0, test_y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer target依照0,1,2..

    BATCH_SIZE = 200      # 批训练的数据个数
    # 先转换成 torch 能识别的 Dataset
    torch_dataset = Data.TensorDataset(train_input_x, train_target_y)

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )

    net = Net(n_feature=200, n_hidden=100, n_output=2)

    net2 = torch.nn.Sequential(
    torch.nn.Linear(200, 150),
    # torch.nn.Tanh(),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    # torch.nn.Tanh(),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 2)
)

    
    Learning_rate = 0.005       # 学习率

    # optimizer 是训练的工具
    optimizer = torch.optim.SGD(net2.parameters(), lr=Learning_rate)  # 传入 net 的所有参数, 学习率
    # 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
    # 但是预测值是2D tensor (batch, n_classes)
    loss_func = torch.nn.CrossEntropyLoss()

    for t in range(80):
        net2.train(mode=True)
        if t % 10 == 0 and t != 0:
            Learning_rate = Learning_rate * 0.8
            # print(Learning_rate)
            if Learning_rate < 0.001:
                Learning_rate = 0.001

            adjust_learning_rate(optimizer,Learning_rate)

        AllLoss = 0.0
        All_train_correct = 0.0
        count = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            out = net2(batch_x)     # 喂给 net2 训练数据 x, 输出分析值
            train_correct = compute_accuracy(out, batch_y)       # 計算正確率
            All_train_correct += train_correct
            # print(out)
            train_loss = loss_func(out, batch_y)     # 计算两者的误差
            AllLoss += train_loss.item()
            count += 1
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            train_loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net2 的 parameters 上
        
        Average_loss = AllLoss/count
        Average_train_correct = All_train_correct/count

        # 測試模式
        net2.eval()
        out = net2(test_input_x)     # 喂给 net2 训练数据 x, 输出分析值
        test_correct = compute_accuracy(out, test_target_y)       # 計算正確率
        # print(out)
        test_loss = loss_func(out, test_target_y)     # 计算两者的误差

        print('第' + str(t+1) + '次' + '訓練模式，loss為:' + str(Average_loss) + '正確率為' + str(Average_train_correct) + '，測試模式，loss為:' + str(test_loss.item()) + '正確率為' + str(test_correct) + '，loss相差' + str(abs(Average_loss-test_loss.item())))
    # torch.save(net2, 'net.pkl')  # 保存整个网络

# 動態調整學習率，參考網站:http://www.spytensor.com/index.php/archives/32/
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden, 10)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(10, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden2(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

# 計算正確值，參考網站:https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/357275/
def compute_accuracy(y_pred, y_target):
    # 過了一道 softmax 的激勵函式後的最大概率才是預測值 
    # torch.max既返回某個維度上的最大值，同時返回該最大值的索引值
    prediction = torch.max(F.softmax(y_pred), 1)[1] # 在第1維度取最大值並返回索引值 
    # print(prediction)   # tensor([1, 1, 1,  ..., 1, 1, 1])
    pred_y = prediction.data.numpy().squeeze() 
    target_y = y_target.data.numpy() 
    accuracy = sum(pred_y == target_y)
    return accuracy/ len(pred_y) * 100

def predict_sent(sent):
    word_list = list(jieba.cut(sent))
    print(word_list)
    # word2vec取向量
    model = models.Word2Vec.load('word2vec.model')
    count = np.zeros(200)   # 紀錄單詞出現在word2vec的次數
    AllVector = np.zeros(200)
    for word in word_list:
        sent_T_F = 0
        if word != '\n':
            try: 
                vector = model[word]    # 被我用來判斷word是否存在word2vec中
                AllVector = np.add(AllVector, vector)
                count = np.add(count, np.ones(200))
                if sent_T_F == 0:
                    sent_T_F = 1
            except:
                continue
        
    if sent_T_F:
        AllVector = np.divide(AllVector, count)   # 取平均

    sent_vector = torch.FloatTensor(AllVector)
    # print(sent_vector)

    # doc2vec取向量
    # model_dm = Doc2Vec.load('sent2vec.model')
    # inferred_vector_dm = model_dm.infer_vector(word_list)         # 將句子透過模型獲得向量
    # print(inferred_vector_dm)
    # sent_vector = torch.FloatTensor(inferred_vector_dm)
    # print(sent_vector)

    net2 = torch.load('net.pkl')
    out = net2(sent_vector)
    x = F.softmax(out)
    prediction = x.tolist()
    # print(prediction)
    negative = prediction[0]
    positive = prediction[1]

    if positive > negative:
        print('正向')
    else:
        print('反向')

    print(positive)
    print(negative)

# 載入向量
def LoadVector(filepath, model, VectorList):
    f = open(filepath, 'r', encoding='UTF-8')
    line = f.readline()
    while line:
        count = np.zeros(200)   # 紀錄單詞出現在word2vec的次數
        AllVector = np.zeros(200)
        sent_list = line.split(' ')
        sent_T_F = 0
        for sent in sent_list:
            if sent != '\n':
                try: 
                    vector = model[sent]    # 被我用來判斷word是否存在word2vec中
                    AllVector = np.add(AllVector, vector)
                    count = np.add(count, np.ones(200))
                    if sent_T_F == 0:
                        sent_T_F = 1
                except:
                    continue
        if sent_T_F:
            AllVector = np.divide(AllVector, count)   # 取平均
        VectorList.append(AllVector)
        line = f.readline()
    f.close()

if __name__ == '__main__':
    path = os.getcwd()  # 當前路徑
    
    # 引入繁體中文詞庫
    jieba.initialize(os.path.join(path, 'jieba_dict/dict.txt.big'))
    # 載入自訂的詞庫(可以多個載入，前面的資料會保留
    jieba.load_userdict(os.path.join(path, 'jieba_dict/mydict'))
    # jieba.load_userdict(os.path.join(path, 'jieba_dict/ptt.txt'))
    # jieba.load_userdict(os.path.join(path, 'jieba_dict/wiki.dict.txt'))
    # jieba.load_userdict(os.path.join(path, 'jieba_dict/attractions.dict.txt'))
    # jieba.load_userdict(os.path.join(path, 'jieba_dict/dcard.dict.txt'))
    # jieba.load_userdict(os.path.join(path, 'jieba_dict/zh_translate_en.dict'))

    # doc2vec模型，參考網站:https://www.itread01.com/content/1548290529.html, https://radimrehurek.com/gensim/models/doc2vec.html
    # x_train = cut_sentence()
    
    
    # test()
    useInt()
    # while True:
    #     sentence = input('請輸入你要測試的句子:')
    #     predict_sent(sentence)
