import torch
import torch.nn as nn
from attention import *


class BottleNeckFusion(nn.Module):
    def __init__(self):
        super(BottleNeckFusion, self).__init__()
        self.embed_dim = 32
        self.seq = 4
        self.layer_unimodal = 1
        self.layer_multimodal = 2
        self.audio_linear = nn.Linear(74, self.embed_dim)
        self.visual_linear = nn.Linear(35, self.embed_dim)
        self.text_linear = nn.Linear(768, self.embed_dim)
        self.audio_tranformer = TransformerEncoder()
        self.visual_transformer = TransformerEncoder()
        self.text_transformer = TransformerEncoder()
        self.bottle_neck = TransformerEncoder()



    def forward(self, audio, visual, text):
        return 0
