import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_layers, id_embedding_size, attention_size):
        super(Attention, self).__init__()

        self.id_embedding_size = id_embedding_size
        self.attention_size = attention_size

        self.review_linear = nn.Linear(hidden_layers, attention_size, bias=True)
        self.id_linear = nn.Linear(id_embedding_size, attention_size, bias=False)
        self.att_linear = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, reviews_embed, ids_embed):
        """
        :param reviews_embed: [batch, review_num, hidden_layers]   [batch, user_review_num, hidden_size]
        :param ids_embed: [batch, user_review_num, id_embedding_size]
        :return:
        """
        review_linear_out = self.review_linear(reviews_embed) # [batch, review_num, attention_size]
        id_linear_out = self.id_linear(ids_embed) # [batch, review_num, attention_size]
        output = review_linear_out + id_linear_out
        output = self.relu(output)
        output = self.att_linear(output) # [batch, review_num, 1]
        output = self.softmax(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, review_hidden_size, da=100, r=10):
        super(SelfAttention, self).__init__()

        self.review_hidden_size = review_hidden_size
        self.da = da
        self.r = r
        self.linear1 = nn.Linear(review_hidden_size, da)
        self.linear2 = nn.Linear(da, r)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, review_embedding):
        # review_embedding: [batch*review_num, review_len, review_hidden_size]
        linear_output = self.linear1(review_embedding)  # [batch*review_num, review_len, da]
        attention = self.linear2(linear_output)  # [batch*review_num, review_len, r]
        attention = attention.expand(attention.size(0), attention.size(1), 1536)
        #attention = self.softmax(attention)

        return attention


class SoftAttention(nn.Module):
    def __init__(self, review_num_filters, id_embedding_size, soa_size=100):
        super(SoftAttention, self).__init__()

        self.linear1 = nn.Linear(id_embedding_size, soa_size)
        self.linear2 = nn.Linear(soa_size, review_num_filters)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, review_embedding, id_embedding):
        # review_embedding: [batch, review_num_filters]
        # id_embedding: [batch, id_embedding_size]

        soft_att = self.linear1(id_embedding) # [batch, soa_size]
        soft_att = self.linear2(soft_att) # [batch, review_num_filters]
        # soft_att = self.softmax(soft_att)

        output = soft_att * review_embedding # [batch, review_num_filters]
        return output

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
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
