'''
实现selfattention和muiltiheadattention机制
'''
import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, ):
        super(AttentionModule, self).__init__()
        #w1和为w2是用来计算注意力得分的参数
        self.Wq = torch.randn(256 , 128)
        self.Wk = torch.randn(256 , 128)
        self.Wv = torch.randn(256 , 128)
        nn.init.uniform_(self.Wq)
        nn.init.uniform_(self.Wk)
        nn.init.uniform_(self.Wv)
        pass

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
        Q = torch.matmul(x, self.Wq)#[N,T,H]*[H,]->[N,T,H]
        K = torch.matmul(x, self.Wk)#[N,T,H]*[H,H]->[N,T,H]
        V = torch.matmul(x, self.Wv)#[N,T,H]*[H,H]->[N,T,H]
        return Q, K, V

    def self_attention(self ,x ):
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


if __name__ == '__main__':
    attention = AttentionModule()
    x = torch.randn(3, 10, 256)#[N,T,H]
    attention.self_attention(x)
