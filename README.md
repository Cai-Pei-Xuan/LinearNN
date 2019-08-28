# LinearNN
神經網路的應用

* 將外賣評論資料集切割成正負向，各向訓練3000筆測試1000筆
  * [外賣評論資料集來源](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k)
* 將句子做斷詞，並透過word2vec取得每個詞的向量，相加後再平均(如果有一個詞沒有向量，取平均時會減掉)
* 使用DataLoader來訓練模型
* 使用神經網路來訓練模型，詳細的參數在報告的ppt中
