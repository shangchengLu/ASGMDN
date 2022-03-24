import torch.nn as nn
import copy

from modules.encoder_decoder import SublayerConnection
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        temp_s = self.sublayer[0]
        temp_s_ = self.sublayer[1]
        temp_s__ = self.sublayer[2]
        # 注：多出来的这一层Attention子层(代码中是src_attn)实现和Self-Attention是一样的，只不过src_attn的Query来自于前层Decoder的输出
        # 但是Key和Value来自于Encoder最后一层的输出(代码中是memory)；而Self-Attention的Q、K、V则均来自前层的输出。
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
