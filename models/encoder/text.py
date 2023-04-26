from transformers import BertModel, BertConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch import nn

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        if config.args.PEFT :
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )

            self.model = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
            self.model.requires_grad_(False)
            print("Frozen all the LLM")

    def forward(self, x):
        return self.model(x)['input']

