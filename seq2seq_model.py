import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, ):
        super(Seq2Seq, self).__init__()
        self.embeding_layer = nn.Embedding(27, 128)
        self.encoder_layer = nn.LSTM(128, 256, num_layers=1, bidirectional=False , batch_first=True)
        self.decoder_layer = nn.LSTM(128, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(256, 27)
        self.loss = nn.CrossEntropyLoss()

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

    # @torch.no_grad()
    def predict(self , x , max_length :int = 30):
        embeding_x = self.embeding_layer(x)#[N,1]->[N,1,E]
        encode_x, (h_e, c_e) = self.encoder_layer(embeding_x)#encode_x[N,1,H] h_e[1,N,H] c_e[1,N,H]
        #解码器,需要构建一个初始的输入，这里就作为SOS,此外需要循环输出，每次将上一次的输出作为下一次的输入，直到输出最大长度或者输出EOS
        input_x = torch.tensor([0]*x.size(0), dtype=torch.long).unsqueeze(1).to(x.device)#假设0是SOS,[N,1],decode_x[0] = GO
        #这里需要一个循环
        outputs = []
        for i in range(max_length):
            if len(outputs) > 0 and outputs[-1] == 99:  # 先判断列表是否为空和最后一个元素是否为EOS
                break
            else:
                input_x = self.embeding_layer(input_x)  # [N,1,E]
                input_x, (h_d, c_d) = self.decoder_layer(input_x, (h_e, c_e))#input_x[N,1,H]
                #x现在要过全连接层
                out = self.fc(input_x)#[N,1,27]
                out = torch.argmax(out, dim=-1)#[N,1]
                outputs.append(out.item())
                #将out作为下一次的输入
                input_x = out
                h_e, c_e = h_d, c_d
        return outputs

if __name__ == '__main__':
    x = ["A", "B" , "C", "D"]
    y = ["B", "C", "D", "E"]
    encode_x = torch.tensor([1, 2, 3, 4],dtype = torch.long)#["A", "B" , "C", "D"]
    decode_x = torch.tensor([0, 2, 3, 4, 5] ,dtype = torch.long)#["<GO>", "B", "C", "D", "E"]
    decode_y = torch.tensor([2, 3, 4, 5, 0],dtype = torch.long)#["B", "C", "D", "E", "<eos>"]
    Seq2Seq_test = Seq2Seq()
    # loss = Seq2Seq_test(encode_x.unsqueeze(0), decode_x.unsqueeze(0), decode_y.unsqueeze(0))
    # print(loss)
    outputs = Seq2Seq_test.predict(encode_x.unsqueeze(0))
    print(outputs)


