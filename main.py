from torch import nn
import torch

class BertEncoder(nn.Module):

    def __init__(self, n_layer=12, d_model=768, n_head=12):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)

    def forward(self, x):
        return self.encoder(x)

if __name__=="__main__":

    bert_encoder = BertEncoder()
    src = torch.rand(10, 32, 768)
    out = bert_encoder(src)
    print(out.shape)