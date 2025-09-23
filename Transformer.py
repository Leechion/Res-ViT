# 导入必要的库
from torch import nn  # 用于定义神经网络层
from transformers import AutoConfig  # 导入自动配置类，用于获取预训练模型的配置
from transformers import AutoTokenizer  # 导入自动分词器类，用于文本的分词和编码
import torch
import torch.nn.functional as F  # 导入PyTorch的nn.functional模块，包含了许多神经网络操作的函数
from math import sqrt  # 用于计算平方根

# 指定预训练的BERT模型
model_ckpt = "bert-base-uncased"

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)  # 从预训练模型加载分词器

# 准备输入文本
text = "time flies like an arrow"

# 使用分词器处理文本，返回特殊的tensor格式
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # 不添加特殊标记
print(inputs.input_ids)  # 打印输入的token ids

# 获取模型的配置
config = AutoConfig.from_pretrained(model_ckpt)  # 从预训练模型加载配置

# 创建一个嵌入层，用于将token ids转换为词向量
# vocab_size是词汇表的大小，hidden_size是嵌入向量的维度
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)  # 打印嵌入层

# 将输入的token ids通过嵌入层转换为词向量
inputs_embeds = token_emb(inputs.input_ids)  # 调用嵌入层的forward方法
print(inputs_embeds.size())  # 打印词向量的尺寸

# 假设inputs_embeds是之前步骤中得到的词嵌入向量，这里同时作为查询（Q）、键（K）和值（V）
Q = K = V = inputs_embeds  # 这里的Q, K, V是相同的词嵌入向量，但在实际应用中它们可能来自不同的输入

# 获取键（K）的维度大小，即键的嵌入维度
dim_k = K.size(-1)  # -1表示最后一个维度，这里是嵌入向量的维度

# 计算注意力分数
'''
为了计算Q和K的点积，我们需要调整K的维度，以便它与Q的维度兼容。原始的K矩阵具有形状(batch_size, sent_len, emb_dim)，
为了执行矩阵乘法，我们需要将K的第二个和第三个维度交换，这里我们可以使用转置操作transpose(1, 2)将K的维度变为(batch_size, emb_dim, sent_len)。

现在，Q和K的形状都是(batch_size, sent_len, emb_dim)，我们可以执行矩阵乘法。

Q 的形状：(batch_size, sent_len, emb_dim)

K 的转置的形状：(batch_size, emb_dim, sent_len)

注意力分数的形状：(batch_size, sent_len, sent_len)
'''
# 每个分数表示一个查询向量与所有键向量之间的相似度
scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(dim_k)  # 计算得到的分数矩阵并缩放
# 这个矩阵中的每个元素表示一个查询向量与一个键向量之间的相似度分数。

# 打印注意力分数的形状
print(scores.size())  # 应该输出(batch_size, sent_len, sent_len)
print(scores)


# 具体来说，scores 的每个元素 scores[b, i, j] 表示第 b 个批次中第 i 个查询向量与第 j 个键向量之间的相似度分数。


# 定义Scaled Dot-product Attention函数
def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    # 获取查询（query）的最后一个维度大小，即键（key）的维度
    dim_k = query.size(-1)

    # 计算查询和键的点积，并缩放，得到未归一化的注意力分数
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)

    # 如果提供了查询掩码（query_mask）和键掩码（key_mask），则计算掩码矩阵
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    else:
        # 如果没有提供掩码，则使用之前传入的掩码（如果有的话）
        mask = mask

    # 如果存在掩码，则将分数矩阵中与掩码对应位置为0的分数替换为负无穷
    # 这样在应用softmax时，这些位置的权重会接近于0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))

    # 使用softmax函数对分数进行归一化，得到注意力权重
    weights = F.softmax(scores, dim=-1)

    # 计算加权后的输出，即将注意力权重与值（value）相乘
    # 这里的输出是经过注意力加权后的值向量，用于下游任务
    return torch.bmm(weights, value)


# 定义AttentionHead类，继承自nn.Module
class AttentionHead(nn.Module):
    # 初始化函数
    def __init__(self, embed_dim, head_dim):
        super().__init__()  # 调用基类的初始化方法
        # 定义线性层，用于将输入的词嵌入向量转换为查询（q）、键（k）和值（v）向量
        # embed_dim是输入嵌入的维度，head_dim是每个头输出的维度
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    # 前向传播函数
    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        # 调用scaled_dot_product_attention函数，传入通过线性层转换后的查询、键和值
        # 同时传入可选的掩码参数
        attn_outputs = scaled_dot_product_attention(
            self.q(query),  # 经过查询线性层转换的query
            self.k(key),  # 经过键线性层转换的key
            self.v(value),  # 经过值线性层转换的value
            query_mask,  # 查询掩码
            key_mask,  # 键掩码
            mask  # 已有的掩码（如果有的话）
        )
        # 返回注意力机制的输出
        return attn_outputs
