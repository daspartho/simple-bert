from torch import nn
import torch

class BertEncoder(nn.Module):

    def __init__(self, n_layer=12, d_model=768, n_head=12, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model, activation="gelu", dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)

    def forward(self, x, mask):
        return self.encoder(x, mask=mask)

class BertEmbedding(nn.Module):

    def __init__(self, vocab_size=30522, max_len=512, n_segment=2, d_model=768, dropout=0.1, pad_idx=0):
        super().__init__()
        self.token = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx)
        self.position = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model, padding_idx=pad_idx)
        self.segment = nn.Embedding(num_embeddings=n_segment, embedding_dim=d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids, position_ids, segment_ids):
        x = self.token(token_ids) + self.position(position_ids) + self.segment(segment_ids)
        return self.dropout(x)

class BertModel(nn.Module):

    def __init__(self, vocab_size=30522, max_len=512, n_segment=2, pad_idx=0, n_layer=12, d_model=768, n_head=12, dropout=0.1):
        super().__init__()
        self.embedding= BertEmbedding(vocab_size, max_len, n_segment, d_model, dropout, pad_idx)
        self.encoder = BertEncoder(n_layer, d_model, n_head, dropout)
        self.n_head = n_head
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, token_ids, position_ids, segment_ids):
        mask = (token_ids > 0).unsqueeze(1).repeat(self.n_head, token_ids.size(1), 1)
        x = self.embedding(token_ids, position_ids, segment_ids)
        x = self.encoder(x, mask)
        return x

class BertLM(nn.Module):

    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.next_sentence_prediction = nn.Sequential(
            nn.Linear(bert_model.d_model, 2),
            nn.LogSoftmax(dim=-1),
        )
        self.masked_lm = nn.Sequential(
            nn.Linear(bert_model.d_model, bert_model.vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, token_ids, position_ids, segment_ids):
        x = self.bert_model(token_ids, position_ids, segment_ids)
        return self.next_sentence_prediction(x[:,0]), self.masked_lm(x)

class BertNextSentencePrediction(nn.Module):

    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.linear = nn.Linear(bert_model.d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, token_ids, position_ids, segment_ids):
        x = self.bert_model(token_ids, position_ids, segment_ids)
        return self.softmax(self.linear(x[:,0]))

class BertMaskedLM(nn.Module):

    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.linear = nn.Linear(bert_model.d_model, bert_model.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, token_ids, position_ids, segment_ids):
        x = self.bert_model(token_ids, position_ids, segment_ids)
        return self.softmax(self.linear(x))

if __name__=="__main__":

    bert_model = BertModel()
    bert_lm = BertLM(bert_model)

    batch_size = 32
    seq_len = 64

    token_ids = torch.randint(30522, size=(batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    segment_ids = torch.randint(2, size=(batch_size, seq_len))

    next_sentence_prediction, masked_lm = bert_lm(token_ids, position_ids, segment_ids)
    print(next_sentence_prediction.shape, masked_lm.shape)