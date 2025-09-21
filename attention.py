'''
实现selfattention和muiltiheadattention机制
'''
import copy

import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    def __init__(self, ):
        super(AttentionModule, self).__init__()
        #w1和为w2是用来计算注意力得分的参数
        self.Wq = torch.nn.Linear(256, 128)
        self.Wk = torch.nn.Linear(256, 128)
        self.Wv = torch.nn.Linear(256, 128)
        nn.init.uniform_(self.Wq.weight)
        nn.init.uniform_(self.Wk.weight)
        nn.init.uniform_(self.Wv.weight)
        self.input_size = self.Wq.weight.shape[1]
        self.hidden_size = self.Wq.weight.shape[0]

    def calc_sorce(self, query, key):
        '''
        计算注意力得分
        :param query:
        :param key:
        :return:
        '''
        score = torch.matmul(query, key.permute(0, 2, 1))#[N,T,aH]*[N,aH,T]->[N,T,T]
        '''
        第一个T是 query 的序列长度（每个 query 对应一行），
        第二个T是 key 的序列长度（每行中的元素对应每个 key 的分数）。
        '''
        score = score / (key.size(-1) ** 0.5)
        score = torch.softmax(score, dim=-1)
        return score

    def creat_QKV(self, x):
        '''
        通过线性变换生成Q,K,V
        :param x:
        :return:
        '''
        Q = self.Wq(x)#[N,T,H]*[H,]->[N,T,H]
        K = self.Wk(x)#[N,T,H]*[H,H]->[N,T,H]
        V = self.Wv(x)#[N,T,H]*[H,H]->[N,T,H]
        return Q, K, V

    def forward(self ,x ):
        '''
        NOTE:假设query, key, value的隐藏层维度都是H
        实现自注意力机制
        :param query:
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        '''
        #1.计算query和key的点积，得到注意力得分
        query ,key, value = self.creat_QKV(x)
        sorce = self.calc_sorce(query, key)
        #2.将注意力得分和value进行加权求和，得到最终的输出
        out = torch.matmul(sorce, value)#[N,T,T]*[N,T,H]->[N,T,H]
        return out

class MulitiHeadAttentionModule(nn.Module):
    def __init__(self, heads , attention_layer):
        '''
        多头注意力机制
        这里的heads是指多头的数量，attention_layer是一个AttentionModule的实例
        多头注意力机制是将多个注意力机制的输出进行拼接，然后通过一个线性变换得到最终的输出
        :param heads:
        :param attention_layer:
        '''
        super(MulitiHeadAttentionModule, self).__init__()
        layers = []
        for _ in range(heads):
            layers.append(copy.deepcopy(attention_layer))#这里需要深拷贝，因为每个头的参数都是独立的，如果不深拷贝的话，多个头的参数会共享
        self.attention_layer = nn.ModuleList(layers)
        self.fc = nn.Linear(attention_layer.hidden_size*heads, attention_layer.input_size)#线性变换，将拼接后的输出映射到原始的维度，方便后续残差连接

    def forward(self, x ):
        out = []
        for layer in self.attention_layer:
            out.append(layer(x))
        out = torch.cat(out, dim=-1)#out[N,T,heads*H]
        out = self.fc(out)
        out = nn.ReLU()(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,heads,attention_layer):
        super(TransformerBlock, self).__init__()
        self.block = MulitiHeadAttentionModule(heads, attention_layer)
        self.fc =nn.Sequential(
            nn.Linear(in_features=attention_layer.input_size, out_features=attention_layer.hidden_size*2),
            nn.ReLU(),
            nn.Linear(in_features=attention_layer.hidden_size*2, out_features=attention_layer.input_size)
        )
        self.norm = nn.LayerNorm(attention_layer.input_size)

    def forward(self, x ):
        out1 = self.block(x)#out[N,T,H]
        out2 = x + out1#残差连接
        out2 = self.norm(out2)#归一化
        out3 = self.fc(out2)#全连接
        out = out2 + out3
        out = self.norm(out)



        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self ,attention_layer ,N , heads):
        super(TransformerEncoderLayer, self).__init__()
        layers = []
        for i in range(N):
            layers.append(TransformerBlock(heads, attention_layer))
        self.encoder_layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoder_layers(x)
        return out

if __name__ == '__main__':
    attention = AttentionModule()
    x = torch.randn(3, 10, 256)#[N,T,H]
    # attention.self_attention(x)
    # model = MulitiHeadAttentionModule(8, attention)
    model = TransformerEncoderLayer(attention, 3, 8)
    out = model(x)
    print(out.shape)
    print(model)