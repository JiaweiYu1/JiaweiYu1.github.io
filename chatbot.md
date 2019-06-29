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

 1. seq2seq
 2. transformer
 3. birnn-seq2seq-greedy
 

 ## 1. seq2seq
 
 sequence-to-sequence(seq2seq)模型，是RNN的变种，是为了解决深度学习中输入和输出的序列长度不同(N vs. M),
 也叫做encoder-decoder模型
 
 基本框架：
 seq2seq作为rnn的变种，由encoder和decoder组成(通常使用LSTM或GRU)，encoder将问题输入，decoder将答案输出，
 首先，我们来看基本的rnn结构，如下图
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/rnn.JPG)
 
 
 上图显示的是基本的n对n的rnn的结构，seq2seq的出现是因为现有的rnn结构无法处理n对m的情况，所以我们分别使用encoder
 和decoder来处理n个输入和m个输出，结构如下：
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/seq2seq_.JPG)
 
 c为中间状态，在decoder过程中要把c输入到每一个hidden layer中，所以它必须包含原始系列中所有信息，当输入信息过长
 时，就会造成精度下降，因此我们引入attention机制，结构如下：
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/seq2seq_attn.JPG)
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/seq2seq_attn_2.JPG)
 
 在tensorflow中的代码如下(不包含attention)
 ```markdown
 class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size, learning_rate, batch_size):
        
        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer,reuse=reuse)
        
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        batch_size = tf.shape(self.X)[0]
        
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        main = tf.strided_slice(self.X, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        decoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, decoder_input)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        _, last_state = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded,
                                          sequence_length=self.X_seq_len,
                                          dtype = tf.float32)
        with tf.variable_scope("decoder"):
            rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells_dec, decoder_embedded, 
                                           sequence_length=self.X_seq_len,
                                           initial_state = last_state,
                                           dtype = tf.float32)
        self.logits = tf.layers.dense(outputs,to_dict_size)
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 ```
 
 我们使用Cornell movie dialogs corpus作为输入数据 [cornell movie corpus](https://github.com/JiaweiYu1/JiaweiYu1.github.io/tree/master/cornell%20movie-dialogs%20corpus)
 在20个epoch之后的训练结果为
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/seq2seq_result.png)
 
 ## 2. transformer
 
 transformer模型也是encoder-decoder结构，如下图
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/transformer_gen.png)
 
 具体的模型结构为下图
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/transformer_de.png)
 
 transformer的encoder部分有N个layer组成，每个layer由两个sub-layer组成，分别是multi-head attention和feed 
 forward, 其中每个sub layer都增加了残差链接和层标准化
 
 ---Attention机制 ([Attention is all your need](https://arxiv.org/abs/1706.03762))
 
 ''attention的机制其实就是一个寻址过程，给定一个人物相关的查询query q, 通过计算key的注意力分布并附加在value上
 从而计算attention value， 这样不需要将所有的N个输入信息都输入到神经网络进行计算， 只需要从X中选择一些和人物相关的信
 息输入给神经网络''
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/attention.png)
 
 1. Self-attention
 
 self-attention就是取Q, K, V相同，是word embedding和position embedding相加的结果，它的线形变换采用了scaled 
 dot-product attention, 公式为：
 
 
 #####                Attention(Q,K,V) = softmax(Q*K<sup>T</sup>/sqrt(d<sub>k</sub>))V
 
 
 2. Multi-head self attention
 
 multi-head self attention相当于h个不同的self-attention的集成，将输入数据X分别输入到h个self-attention当中，得到h
 个加权后的特征矩阵Z<sub>i</sub>,之后将h个特征矩阵拼接成一个大的特征矩阵，之后经过一层全联接层之后输出Z
 
 3. Position-wise feed forward networks
 
 Position-wise feed forward networks实际上就是一个多层感知机网络，它包括两层dense,第一层的激活函数为ReLU，第二层为
 一个线形激活函数，公式为
 
 
 #####                PWFFN(Z) = max(0, W<sub>1</sub>Z+b<sub>1</sub>)W<sub>2</sub>+b<sub>2</sub>
 
 
 4. Positional encoding
 
 Positional encoding是对于数据的预处理使模型能够对顺序序列进行处理，transformer将positional encoding后的数据和
 embedding的数据相加, positional encoding的方法可以为用不同频率的sine和cosine函数直接进行计算，公式为
 
 
 #####               PE(pos, 2i) = sin(pos/10000<sup>2i/d<sup>model</sup></sup>)
 #####               PE(pos, 2i+1) = cos(pos/10000<sup>2i/d<sup>model</sup></sup>)
 
 
 transformer在tensorflow中的实现代码为：
 
 ```markdown
 def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    
    outputs = gamma * normalized + beta
    return outputs


def multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads):
    
    T_q = tf.shape(queries)[1]                                      
    T_k = tf.shape(keys)[1]                  

    Q = tf.layers.dense(queries, num_units, name='Q')                              
    K_V = tf.layers.dense(keys, 2*num_units, name='K_V')    
    K, V = tf.split(K_V, 2, -1)        

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                    
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                      

    align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                      
    align = align / np.sqrt(K_.get_shape().as_list()[-1])                 

    paddings = tf.fill(tf.shape(align), float('-inf'))                   

    key_masks = k_masks                                                 
    key_masks = tf.tile(key_masks, [num_heads, 1])                       
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])            
    align = tf.where(tf.equal(key_masks, 0), paddings, align)       

    if future_binding:
        lower_tri = tf.ones([T_q, T_k])                                          
        lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()  
        masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(align)[0], 1, 1]) 
        align = tf.where(tf.equal(masks, 0), paddings, align)                      
    
    align = tf.nn.softmax(align)                                            
    query_masks = tf.to_float(q_masks)                                             
    query_masks = tf.tile(query_masks, [num_heads, 1])                             
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])            
    align *= query_masks                                                           
          
    outputs = tf.matmul(align, V_)                                                 
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)             
    outputs += queries                                                             
    outputs = layer_norm(outputs)                                                 
    return outputs


def pointwise_feedforward(inputs, hidden_units, activation=None):
    outputs = tf.layers.dense(inputs, 4*hidden_units, activation=activation)
    outputs = tf.layers.dense(outputs, hidden_units, activation=None)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


def learned_position_encoding(inputs, mask, embed_dim):
    T = tf.shape(inputs)[1]
    outputs = tf.range(tf.shape(inputs)[1])                # (T_q)
    outputs = tf.expand_dims(outputs, 0)                   # (1, T_q)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])   # (N, T_q)
    outputs = embed_seq(outputs, T, embed_dim, zero_pad=False, scale=False)
    return tf.expand_dims(tf.to_float(mask), -1) * outputs


def sinusoidal_position_encoding(inputs, mask, repr_dim):
    T = tf.shape(inputs)[1]
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1]) * tf.expand_dims(tf.to_float(mask), -1)


def label_smoothing(inputs, epsilon=0.1):
    C = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / C)


class Chatbot:
    def __init__(self, size_layer, embedded_size, from_dict_size, to_dict_size, learning_rate,
                 num_blocks = 2,
                 num_heads = 8,
                 min_freq = 50):
        self.X = tf.placeholder(tf.int32,[None,None])
        self.Y = tf.placeholder(tf.int32,[None,None])
        
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        
        encoder_embedding = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embedding = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        
        def forward(x, y):
            encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, x)
            en_masks = tf.sign(x)
            encoder_embedded += sinusoidal_position_encoding(x, en_masks, embedded_size)
        
            for i in range(num_blocks):
                with tf.variable_scope('encoder_self_attn_%d'%i,reuse=tf.AUTO_REUSE):
                    encoder_embedded = multihead_attn(queries = encoder_embedded,
                                             keys = encoder_embedded,
                                             q_masks = en_masks,
                                             k_masks = en_masks,
                                             future_binding = False,
                                             num_units = size_layer,
                                             num_heads = num_heads)

                with tf.variable_scope('encoder_feedforward_%d'%i,reuse=tf.AUTO_REUSE):
                    encoder_embedded = pointwise_feedforward(encoder_embedded,
                                                    embedded_size,
                                                    activation = tf.nn.relu)
            
            decoder_embedded = tf.nn.embedding_lookup(decoder_embedding, y)
            de_masks = tf.sign(y)
            decoder_embedded += sinusoidal_position_encoding(y, de_masks, embedded_size)
            
            for i in range(num_blocks):
                with tf.variable_scope('decoder_self_attn_%d'%i,reuse=tf.AUTO_REUSE):
                    decoder_embedded = multihead_attn(queries = decoder_embedded,
                                         keys = decoder_embedded,
                                         q_masks = de_masks,
                                         k_masks = de_masks,
                                         future_binding = True,
                                         num_units = size_layer,
                                         num_heads = num_heads)
                
                with tf.variable_scope('decoder_attn_%d'%i,reuse=tf.AUTO_REUSE):
                    decoder_embedded = multihead_attn(queries = decoder_embedded,
                                         keys = encoder_embedded,
                                         q_masks = de_masks,
                                         k_masks = en_masks,
                                         future_binding = False,
                                         num_units = size_layer,
                                         num_heads = num_heads)
                
                with tf.variable_scope('decoder_feedforward_%d'%i,reuse=tf.AUTO_REUSE):
                    decoder_embedded = pointwise_feedforward(decoder_embedded,
                                                    embedded_size,
                                            activation = tf.nn.relu)
            
            return tf.layers.dense(decoder_embedded, to_dict_size, reuse=tf.AUTO_REUSE)
        
        self.training_logits = forward(self.X, decoder_input)
        
        def cond(i, y, temp):
            return i < 2 * tf.reduce_max(self.X_seq_len)
        
        def body(i, y, temp):
            logits = forward(self.X, y)
            ids = tf.argmax(logits, -1)[:, i]
            ids = tf.expand_dims(ids, -1)
            temp = tf.concat([temp[:, 1:], ids], -1)
            y = tf.concat([temp[:, -(i+1):], temp[:, :-(i+1)]], -1)
            y = tf.reshape(y, [tf.shape(temp)[0], 2 * tf.reduce_max(self.X_seq_len)])
            i += 1
            return i, y, temp
        
        target = tf.fill([batch_size, 2 * tf.reduce_max(self.X_seq_len)], GO)
        target = tf.cast(target, tf.int64)
        self.target = target
        
        _, self.predicting_ids, _ = tf.while_loop(cond, body, 
                                                  [tf.constant(0), target, target])
        self.logits = forward(self.X, self.Y)
        self.k = tf.placeholder(dtype = tf.int32)
        p = tf.nn.softmax(self.logits)
        self.topk_logprobs, self.topk_ids = tf.nn.top_k(tf.log(p), self.k)
        
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 ```
 
 在20个epoch之后的训练结果为：
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/transfomer_result.png)
 
 
 ## 3. birnn-seq2seq-greedy
 
 我们之前的seq2seq模型使用的是单向RNN构成，由于单向的RNN在时序上处理序列，所以会忽略未来的上下文的信息，因此，我们使用
 双向RNN模型(birnn)。基本思想是每一个序列向前或者向后分别是两个RNN，两个都连着一个输出层，如下图：
 
 ![Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/birnn_seq2seq.jpg)
 
 实现代码如下：
 
 ```markdown
 class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size, 
                 from_dict_size, to_dict_size, learning_rate, 
                 batch_size, dropout = 0.5, beam_width = 15):
        
        def lstm_cell(size, reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(),
                                           reuse=reuse)
        
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        # encoder
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = lstm_cell(size_layer // 2),
                cell_bw = lstm_cell(size_layer // 2),
                inputs = encoder_embedded,
                sequence_length = self.X_seq_len,
                dtype = tf.float32,
                scope = 'bidirectional_rnn_%d'%(n))
            encoder_embedded = tf.concat((out_fw, out_bw), 2)
        
        bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
        bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
        self.encoder_state = tuple([bi_lstm_state] * num_layers)
        
        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(num_layers))
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        # decoder
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        decoder_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)])
        dense_layer = tf.layers.Dense(to_dict_size)
        training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embeddings, decoder_input),
                sequence_length = self.Y_seq_len,
                time_major = False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cells,
                helper = training_helper,
                initial_state = self.encoder_state,
                output_layer = dense_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = decoder_embeddings,
                start_tokens = tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
                end_token = EOS)
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cells,
                helper = predicting_helper,
                initial_state = self.encoder_state,
                output_layer = dense_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
        self.training_logits = training_decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.sample_id
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 ```
 
 在20个epoch之后的训练结果为：
 [!Image](https://github.com/JiaweiYu1/JiaweiYu1.github.io/blob/master/images/birnn_result_train.png)
 
 训练集的BLEU score为0.826，测试集的BLEU score为0.702
 
 
 # 总结
 
最初的seq2seq我们使用最基本的RNN模型，并且没有加入attention，虽然我们输入的数据长度十分短，但结果依然不是很好。之后使用transformer模型，我们引入multi-head self attention，结果有所提高。最后我们使用以LSTM为基础的birnn-seq2seq模型，结果又有所提高。
 
 
 
 
 
 
 
 
 Reference:
 
 1.[NLP Models tensorflow](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/chatbot)
 
 2.[真正的完全图解seq2seq attention模型](https://zhuanlan.zhihu.com/p/40920384)
 
 3.[深度学习：transformer模型](https://blog.csdn.net/pipisorry/article/details/84946653)
