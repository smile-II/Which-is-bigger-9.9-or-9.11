import torch
import torch.nn as nn
from transformers import BertModel

class Discriminator(nn.Module):
    """
    判别器类，用于判断输入序列的真实性。
    
    该类基于BERT模型，添加了一个线性分类器和一个sigmoid激活函数，用于输出判别结果。
    
    参数:
    bert_model_name: 字符串，指定预训练的BERT模型名称，默认为'bert-base-uncased'。
    """
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(Discriminator, self).__init__()
        # 初始化预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        # 添加一个线性分类器，用于从BERT的输出中提取判别特征
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        # 添加sigmoid激活函数，用于将分类器的输出转换为0到1之间的概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """
        判别器的前向传播方法。
        
        参数:
        input_ids: LongTensor，形状为(batch_size, sequence_length)，表示输入的序列编号。
        attention_mask: LongTensor，形状为(batch_size, sequence_length)，表示输入序列的注意力掩码。
        
        返回:
        sigmoid(logits): Tensor，形状为(batch_size, )，表示每个输入序列是真实还是伪造的概率。
        """
        # 通过BERT模型获取输入序列的表示和池化输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 提取池化输出，通常用于分类任务
        pooled_output = outputs.pooler_output
        # 通过线性分类器将池化输出转换为判别结果的logits
        logits = self.classifier(pooled_output)
        # 应用sigmoid激活函数，得到0到1之间的概率输出
        return self.sigmoid(logits)