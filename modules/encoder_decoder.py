from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
from modules.attention import SelfAttention
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.attention import MultiHeadedAttention
from .att_model import pack_wrapper, AttModel
from pytorch_transformers import BertModel, BertConfig, BertTokenizer

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def add_norm(x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    pass
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm, attn, ff):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        self.attn = attn
        self.ff = ff
        self.encoder_ = encoder



    def forward(self, src, top_id, tgt, src_mask, tgt_mask):
        Flag = True
        T_hat = self.encode(src, top_id, src_mask, Flag)
        Flag = False
        I_hat = self.encode(src, T_hat, src_mask, Flag)
        I_hat = self.encoder.norm(T_hat + I_hat)
        return self.decode(I_hat, src_mask, tgt, tgt_mask)  # decode的hidden_states


    def encode(self, src, src_id, src_mask, Flag):
            if Flag == True:
                return self.encoder(self.src_embed(src), self.tgt_embed(src_id), src_mask, Flag)
            elif Flag == False:
                return self.encoder(self.src_embed(src), src_id, src_mask, Flag)
            else:
                return self.encoder(self.src_embed(src), src_id, src_mask, Flag)
    def decode(self, hidden_states, src_mask, tgt, tgt_mask):


        # memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        #
        # #memory = self.rm(self.tgt_embed(tgt), memory)  原来的
        #
        # tgt_embed = self.tgt_embed(tgt)
        emb = 5
        memory = self.rm(self.tgt_embed(tgt))

        #memory = self.bert(tgt)
        #temp_memory = self.rm(memory)
        #temp_memory = self.rm(self.tgt_embed(tgt))
        #temp_memory = self.    rm(self.tgt_embed(tgt), tgt, memory)



        #temp_memory = torch.randn(2,34,512)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, src_id, mask, Flag):
        for layer in self.layers:
            x = layer(x, src_id, mask, Flag)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, src_id, mask, Flag):
        if Flag == True :
            x = self.sublayer[0](x, lambda x: self.self_attn(x, src_id, src_id, mask))
            return self.sublayer[1](x, self.feed_forward)
        elif Flag == False:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, src_id, mask))
            return self.sublayer[1](x, self.feed_forward)
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)

        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):

        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)

        return self.sublayer[2](x, self.feed_forward, memory)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):

        return x + self.dropout(sublayer(self.norm(x, memory)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(d_model, d_model))  #rm_d_model 512

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        #   memory 是图片增强了文字之后的文字的特征  == 当前文字特征得到了加强
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)

        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)


        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]


        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ReviewLSTM(nn.Module):
    """
    LSTM 从评论里提取特征
    """

    def __init__(self, embedding_size=300, dropout=0.,
                 hidden_size=512, num_layers=1, bidirectional=False):
        super(ReviewLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)

        self.ndirections = 1 + int(bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        print("我是ReviewLSTM，我被调用了")
    def forward(self, x):
        """
        :param x: [batch, review_num, review_len, embedding_size]
        :return:
        """
        # batch = x.size(0)
        print("我是ReviewLSTM，我被调用了")
        review_num = x.size(0)
        review_len = x.size(1)
        embedding_size = x.size(2)
        temp_x = x
        x = x.view(-1, review_len, embedding_size)  # [batch*review_num, review_len, embedding_size]
        is_equal = temp_x.equal(x)
        # output: [batch*review_num, review_len, hidden_layers*ndirections]
        # hn: [ndirections*num_layers, batch*review_num, hidden_layers]
        output, (hn, cn) = self.lstm(x)
        # [batch*review_num, hidden_layers*ndirections]
        temp_output = output
        #output = output[:, -1]
        # [batch, review_num, hidden_layers*ndirections]
        # output = output.view(x.size(0), review_num, -1)
        return output

class GRUModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bidirectional=True):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, num_layers=num_layers, bidirectional=bidirectional)


    def forward(self, x):

        x, self.hidden = self.gru(x)

        return x
class TextNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('F://xyy//bert-base-uncased//bert-base-uncased//bert_config.json')
        self.textExtractor = BertModel.from_pretrained(
            'F://xyy//bert-base-uncased//bert-base-uncased//pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens):
        output = self.textExtractor(tokens)
        text_embeddings = output[0]
        # output[0](batch size, sequence length, model hidden dimension)
        # temp = self.textExtractor(tokens)
        # temp__ = temp[0]
        # equal_ = text_embeddings.equal(temp__)
        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features

class GRUModelSA(nn.Module):
    """
    LSTM + self-attention从评论里提取特征
    """

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bidirectional=True):
        super(GRUModelSA, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                          num_layers=num_layers, bidirectional=bidirectional)



        self.review_hidden_size = hidden_size * (1 + int(bidirectional))
        self.da = 100
        self.r = 1

        self.review_att = SelfAttention(self.review_hidden_size, self.da, self.r)
        self.mhat = MultiHeadedAttention(8, 1536)
    def forward(self, x):
        """
        :param x: [batch, review_num, review_len, embedding_size]
        :return:
        """
        # batch = x.size(0)
        # review_num = x.size(1)
        review_len = x.size(1)
        embedding_size = x.size(2)

        # [batch*review_num, review_len, embedding_size]
        x = x.view(-1, review_len, embedding_size)

        # output: [batch*review_num, review_len, review_hidden_size]
        # hn: [ndirections*num_layers, batch*review_num, hidden_layers]
        output, hn = self.gru(x)

        review_embedding = output # [batch*review_num, review_len, review_hidden_size]
        att = self.mhat(output,output,output)
        # attention = self.review_att(review_embedding) # [batch*review_num, r, review_len]
        # review_embedding = attention.mul(review_embedding)  # [batch*review_num, r, review_hidden_size]
        # review_embedding = torch.sum(review_embedding, 1) / self.r  # [batch*review_num, review_hidden_size]
        # review_embedding = review_embedding.view(batch, review_num, self.review_hidden_size) # [batch, review_num, review_hidden_size]

        # att = torch.mean(attention, 1) # [batch*review_num, review_len]
        # att = att.view(batch, review_num, review_len) # [batch, review_num, review_len]

        return att
class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)  #模型图中的Z
        next_memory = next_memory + self.mlp(next_memory)  # 模型中的hat Mt



        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward_step(self, input, hidden_states, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        temp_q = input.unsqueeze(1)
        temp_q_ = torch.cat([memory, input.unsqueeze(1)], 1)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)


        next_memory = memory + self.attn(q, k, v)  #模型图中的Z
        next_memory = next_memory + self.mlp(next_memory)  # 模型中的hat Mt



        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory
    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            temp_fs = inputs[:, i]
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def forward(self, inputs, hidden_states, memory):

        outputs = []
        temp_input0 = inputs[0]
        temp_input1 = inputs[1]
        flag = temp_input0.equal(temp_input1)
        for i in range(inputs.shape[1]):
            temp_fs = inputs[:, i]
            temp_____ = hidden_states[:, i]
            memory = self.forward_step(inputs[:, i], hidden_states, memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        #rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        #gru = GRUModel(self.d_model, 768, self.num_layers)
        grusa = GRUModelSA(self.d_model, 768, self.num_layers)
        #bert = TextNet(code_length=1536)
        #lstm = ReviewLSTM(self.d_model, self.dropout, 768, self.num_layers, True)
        #attn_lstm = ReviewLSTMSA(self.d_model, self.dropout, 1024, self.num_layers, True,100,10)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            grusa, attn, ff)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model
        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        #memory = self.model.encode(att_feats, att_masks) 原来的
        memory = self.model.encode(att_feats, att_masks, seq, seq_mask)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) #将提取的特征进行了简单的线性运算

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, top_id, att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, top_id, seq, att_masks, seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
