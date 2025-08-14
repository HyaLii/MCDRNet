from torch import nn
from net.encoder import CDRE
from net.RT import RT

class MCDRNet(nn.Module):
    def __init__(self, opt):
        super(MCDRNet, self).__init__()

        # Restorer
        self.R = RT(opt)

        # Encoder
        self.E = CDRE(opt)

    def forward(self, x_query, x_key):
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            restored = self.R(x_query, inter)

            return restored, logits, labels
        else:
            fea, inter = self.E(x_query, x_query)

            restored = self.R(x_query, inter)

            return restored
