from transformers import BertModel, BertConfig, BertTokenizer
from modules.lora import *
from torch import nn
from config import *

# class TextEncoder(nn.Module):
#     def __init__(self, config):
#         super(TextEncoder, self).__init__()
#         self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
#         self.model = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)
#         if config.args.PEFT :
#             self.PEFTmodel = LoraModel(config=config, model=self.model) # 评估的时候是否需要mergelinear 即将linear 加入bert
#     def forward(self, x):
#         return self.model(x)['input']


sentence = "i love my mom"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input = tokenizer(sentence, max_length=50, add_special_tokens=True, truncation=True, padding='max_length',return_tensors='pt')
print(input)
config = Config()
bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
model = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

PEFTmodel = LoraModel(config=config, model=model)
# print(model(input_ids=torch.LongTensor(input['input_ids']), token_type_ids=torch.LongTensor(input['token_type_ids']),
#                 attention_mask=torch.LongTensor(input['attention_mask'])))
# print(PEFTmodel(input_ids=torch.LongTensor(input['input_ids']), token_type_ids=torch.LongTensor(input['token_type_ids']),
#                 attention_mask=torch.LongTensor(input['attention_mask'])))
for para in PEFTmodel.parameters(): # 返回的每一个元素是一个元组 tuple
    '''
    是一个元组 tuple ,元组的第一个元素是参数所对应的名称，第二个元素就是对应的参数值
    '''
    #print(para[0], '\t', para[1].size())
    print(para.requires_grad)

