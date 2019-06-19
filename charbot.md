# 什么是chatbot

 什么是chatbot? 大多数人把它称作聊天机器人，也可以叫做对话系统。它是NLP和deep learning中的一个热门话题，
 很多公司想要开发一个机器人能够像人类一样进行自然的对话。

# chatbot的实现分类有哪些

 ## 1. 检索 vs. 生成

 检索类chatbot相对于生成类较为简单，实现时需要将一系列的问题与回答(每个问题与回答为一个pair)存储起来，之后
 根据输入和背景对答案进行检索，检索方式可以为根据某种规则进行搜索或者利用机器学习分类器来进行答案匹配来输出结果，
 这种方式不会产生新的文字，只是从已有的数据库中挑选答案

 生成类chatbot并不依赖于已有的答案，而是根据输入而生成新的答案(我们重点放在如何实现生成类chatbot)

 ## 2. 有范围 vs. 无范围

 无范围的chatbot表示你可以在任何情况中使用它，而有范围的chatbot表示只能在有限的范围中使用，例如淘宝问答chatbot
 或者是电影问答chatbot，很明显设计无范围的chatbot要有无范围的要困难一些。

 ## 3. 短文本 vs. 长文本

 短文本对话指输入与输出的文本较短，长文本对话可能包含大量文字，生成回答可能需要保存之前的文字并记忆重点，相对于短文
 本对话来说较难实现

# 生成类chatbot模型

 ### 1. seq2seq
 ### 2. transformer
 ### 3. birnn-seq2seq-greedy
 ### 4. memory network

 ## 1. seq2seq
 
 sequence-to-sequence(seq2seq)模型，是RNN的变种，是为了解决深度学习中输入和输出的序列长度不同(N vs. M),
 也叫做encoder-decoder模型
 
 基本框架：
 seq2seq作为rnn的变种，由encoder和decoder组成(通常使用LSTM或GRU)，encoder将问题输入，decoder将答案输出，
 首先，我们来看基本的rnn结构，如下图
 
 上图显示的是基本的n对n的rnn的结构，seq2seq的出现是因为现有的rnn结构无法处理n对m的情况，所以我们分别使用encoder
 和decoder来处理n个输入和m个输出，结构如下：
 
 c为中间状态，在decoder过程中要把c输入到每一个hidden layer中，所以它必须包含原始系列中所有信息，当输入信息过长
 时，就会造成精度下降，因此我们引入attention机制，结构如下：
 
 在tensorflow中的代码如下
 
 
 ## 2. transformer
 
 transformer模型也是encoder-decoder结构，如下图
 
 transformer的encoder部分有N个layer组成，每个layer由两个sub-layer组成，分别是multi-head attention和feed 
 forward, 其中每个sub layer都增加了残差链接和层标准化
 
 ---Attention机制
 
 (引用)attention的机制其实就是一个寻址过程，给定一个人物相关的查询query q, 通过计算key的注意力分布并附加在value上
 从而计算attention value， 这样不需要将所有的N个输入信息都输入到神经网络进行计算， 只需要从X中选择一些和人物相关的信
 息输入给神经网络
 

 
 
 
