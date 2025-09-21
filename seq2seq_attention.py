import torch
import torch.nn as nn

class Seq2SeqAttenion(nn.Module):
    def __init__(self, ):
        super(Seq2SeqAttenion, self).__init__()
        self.embeding_layer = nn.Embedding(1000, 128)
        self.encoder_layer = nn.LSTM(128, 256, num_layers=1, bidirectional=False , batch_first=True)
        self.decoder_layer = nn.LSTM(128, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(256*2, 1000)
        self.loss = nn.CrossEntropyLoss()
        self.W1 = torch.randn(1,256)
        self.W2 = torch.randn(1,256)
        nn.init.uniform_(self.W1)
        nn.init.uniform_(self.W2)

    def forward(self, encode_x, decode_x, decode_y):
        #embeding
        embeding_x = self.embeding_layer(encode_x)#[N,T1]->[N,T1,E]
        decode_x = self.embeding_layer(decode_x)#[N,T2]->[N,T2,E]
        #编码器
        encode_x, (h_e, c_e) = self.encoder_layer(embeding_x)#这里的encode_x是没有什么作用的,h_e[batch, num_layers*num_directions,hidden_size] c_e[batch, num_layers*num_directions,hidden_size]
        #解码器
        decode_x, (h_d, c_d) = self.decoder_layer(decode_x, (h_e, c_e))#decode_x[N.T2,H]
        #全连接层
        out = self.fc(decode_x)#[N,T2,27]
        out = out.permute(0,2,1)
        #计算损失
        loss = self.loss(out, decode_y)#out[N,T2,27] decode_y[N,T2],注意维度关系 ，种类必须在0维或者1维，所以需要转换一下
        '''Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.'''
        return  loss

    def forward_v1(self, encode_x, decode_x, decode_y,connection_matrix):
        '''
        因为普通的seq2seq模型在处理长序列时，效果并不好，所以引入注意力机制，这样可以让模型在每一步解码时都能关注到输入序列的不同部分。
        这里关注一下attention的实现，这里的attention是加性注意力，这里生成一个关联矩阵.用来在解码器部分进行加权，这个关联矩阵的计算是基于原始数据对生成数据的影响
        类似现在设计的关联矩阵，也就是每一个输入对每一个输出的影响。这里利用了encode_x的输出和connection_matrix进行矩阵乘法，得到一个新的encode_x，然后和decode_x进行拼接，这样就会使得
        每一个对应的每一个时间点对应的输入都能影响到输出。
        connection_matrix[T1,T2]*encode_x(N,T1,H)+(cat)decode_x[N,T2 ,H]然后再用于全连接层，但是这里一定要注意维度的变化
        decode_x[N,T2 ,H]*(connection_matrix[T2,T1]).T -> [N,T1,H] -> [N,T2,H]
        connection_matrix是一个固定的矩阵
        '''
        '''
        缺点是这个关联矩阵是固定的，没有动态调整的机制，可能会导致模型在某些情况下无法很好地捕捉输入和输出之间的关系。
        优点是计算简单，易于实现，适用于一些简单的任务。
        '''

        # embeding
        embeding_x = self.embeding_layer(encode_x)  # [N,T1]->[N,T1,E]
        decode_x = self.embeding_layer(decode_x)  # [N,T2]->[N,T2,E]
        # 编码器
        encode_x, (h_e, c_e) = self.encoder_layer(embeding_x)  # 这里的encode_x是没有什么作用的,h_e[batch, num_layers*num_directions,hidden_size] c_e[batch, num_layers*num_directions,hidden_size]
        # 注意力机制部分
        encode_x= encode_x.permute(0,2,1)#[N,T1,H]->[N,H,T1]
        encode_x =torch.matmul(encode_x, connection_matrix)  # [N,H,T1]*[T1,T2]->[N,H,T2]
        encode_x = encode_x.permute(0,2,1)#[N,H,T2]->[N,T2,H]
        # 解码器
        decode_x, (h_d, c_d) = self.decoder_layer(decode_x, (h_e, c_e))  # decode_x[N.T2,H]
        # 全连接层
        decode_x = torch.cat((decode_x, encode_x), dim=2)
        out = self.fc(decode_x)  # [N,T2,27]
        out = out.permute(0, 2, 1)
        # 计算损失
        loss = self.loss(out, decode_y)  # out[N,T2,27] decode_y[N,T2],注意维度关系 ，种类必须在0维或者1维，所以需要转换一下
        '''Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.'''
        return loss

    def forward_v2(self, encode_x, decode_x, decode_y):
        '''
        这里不需要connection_matrix，而是通过计算encode_x和decode_x之间的相似度来生成一个动态的关联矩阵，这样可以更好地捕捉输入和输出之间的关系。
        解决了connection_matrix是固定的，没有动态调整的机制的问题。
        '''
        embeding_x = self.embeding_layer(encode_x)  # [N,T1]->[N,T1,E]
        decode_x = self.embeding_layer(decode_x)  # [N,T2]->[N,T2,E]
        # 编码器
        encode_x, (h_e, c_e) = self.encoder_layer(embeding_x)  # 这里的encode_x是没有什么作用的,h_e[batch, num_layers*num_directions,hidden_size] c_e[batch, num_layers*num_directions,hidden_size]
        # 解码器
        decode_x, (h_d, c_d) = self.decoder_layer(decode_x, (h_e, c_e))  # decode_x[N.T2,H]
        # 注意力机制部分
        connection_matrix = torch.matmul(decode_x, encode_x.permute(0,2,1)) #[N,T2,H]*[N,H,T1] -> [N,T2,T1]
        connection_matrix = torch.softmax(connection_matrix, dim=-1).permute(0,2,1)#[N,T2,T1]
        encode_x = encode_x.permute(0, 2, 1)  # [N,T1,H]->[N,H,T1]
        encode_x = torch.matmul(encode_x, connection_matrix)  # [N,H,T1]*[T1,T2]->[N,H,T2]
        encode_x = encode_x.permute(0, 2, 1)  # [N,H,T2]->[N,T2,H]
        # 全连接层
        decode_x = torch.cat((decode_x, encode_x), dim=2)
        out = self.fc(decode_x)  # [N,T2,27]
        out = out.permute(0, 2, 1)
        # 计算损失
        loss = self.loss(out, decode_y)  # out[N,T2,27] decode_y[N,T2],注意维度关系 ，种类必须在0维或者1维，所以需要转换一下
        '''Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.'''
        return loss

    def forward_v3(self, encode_x, decode_x, decode_y):
        '''
        这里实现一个加性注意力机制，这里通过计算encode_x和decode_x之间的相似度来生成一个动态的关联矩阵，这样可以更好地捕捉输入和输出之间的关系。
        '''
        embeding_x = self.embeding_layer(encode_x)  # [N,T1]->[N,T1,E]
        decode_x = self.embeding_layer(decode_x)  # [N,T2]->[N,T2,E]
        # 编码器
        encode_x, (h_e, c_e) = self.encoder_layer(embeding_x)  # 这里的encode_x是没有什么作用的,h_e[batch, num_layers*num_directions,hidden_size] c_e[batch, num_layers*num_directions,hidden_size]
        # 解码器
        decode_x, (h_d, c_d) = self.decoder_layer(decode_x, (h_e, c_e))  # decode_x[N.T2,H]
        # 注意力机制部分
        ensorce_x = torch.matmul(encode_x ,self.W1.permute(1,0)) # [N,T1,H]*[H,1] -> [N,T1,1]
        desorce_x =  torch.matmul(decode_x , self.W2.permute(1,0))  # [N,T2,H]*[H,1] -> [N,T2,1]
        #现在需要的是一个关联矩阵，所以需要进行广播机制
        desorce_x = desorce_x.permute(0,2,1)  # [N,T2,1]->[N,1,T2]
        score = ensorce_x + desorce_x  # [N,T1,1]+[N,1,T2] -> [N,T1,T2]
        score = torch.softmax(score, dim=1)
        # connection_matrix = torch.matmul(decode_x, encode_x.permute(0,2,1)) #[N,T2,H]*[N,H,T1] -> [N,T2,T1] 
        # connection_matrix = torch.softmax(connection_matrix, dim=-1).permute(0,2,1)#[N,T2,T1]
        encode_x = encode_x.permute(0, 2, 1)  # [N,T1,H]->[N,H,T1]
        encode_x = torch.matmul( encode_x ,score )  # [N,H,T1]*[T1,T2]->[N,H,T2]
        encode_x = encode_x.permute(0, 2, 1)  # [N,H,T2]->[N,T2,H]
        # 全连接层
        decode_x = torch.cat((decode_x, encode_x), dim=2)
        out = self.fc(decode_x)  # [N,T2,27]
        out = out.permute(0, 2, 1)
        # 计算损失
        loss = self.loss(out, decode_y)  # out[N,T2,27] decode_y[N,T2],注意维度关系 ，种类必须在0维或者1维，所以需要转换一下
        '''Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.'''
        return loss


if __name__ == '__main__':
    # 古典精华凝诗句，现代风貌入画屏
    x = ["古", "典" , "精", "华","凝", "诗", "句"]
    y = ["现", "代", "风", "貌","入", "画", "屏"]
    encode_x = torch.tensor([3,4,5,6,7,8,9],dtype = torch.long)#["古", "典" , "精", "华","凝", "诗", "句"]
    decode_x = torch.tensor([0, 10, 11,12,13, 14, 15,16] ,dtype = torch.long)#["<GO>", "现", "代", "风", "貌","入", "画", "屏"]
    decode_y = torch.tensor([0, 11,12,13, 14, 15,16 , 1],dtype = torch.long)#  ["现", "代", "风", "貌","入", "画", "屏","<eos>"]
    connection_matrix= torch.randn(7,8) #假设有一个随机的关联矩阵
    Seq2Seq_test = Seq2SeqAttenion()
    loss = Seq2Seq_test.forward_v3(encode_x.unsqueeze(0), decode_x.unsqueeze(0), decode_y.unsqueeze(0))
    print(loss)
    # outputs = Seq2Seq_test.predict(encode_x.unsqueeze(0))
    # print(outputs)


