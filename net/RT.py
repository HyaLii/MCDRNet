import math
import torch
import torch.nn as nn
from .deform_conv import DCN_layer
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta

class FCA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(FCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        #self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=k, padding=int(k / 2)),
            nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=3, padding=1)
        )
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input * out

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class ARM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(ARM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(self.channels_in, self.channels_out, kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out)

        self.fc_att = FCA(channel=channels_out)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter)
        out = dcn_out + sft_out
        out = self.fc_att(out)
        out = x + out

        return out


class L_Block(nn.Module):

    def __init__(self, kernel_size=7):
        super(L_Block, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))

        self.conv5x5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=3)

        self.conv_4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu_4_2 = nn.LeakyReLU(0.2, True)

    def forward(self, x):


        x = self.conv_4_2(x)
        x = self.relu_4_2(x)

        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        conv7x7_out = self.conv7x7(self.conv5x5(x))

        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att1 = torch.sigmoid(conv)
        att = conv7x7_out * att1
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x

class G_Block(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(G_Block, self).__init__()
        midplanes = int(outplanes // 2)

        self.conv_4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu_4_1 = nn.LeakyReLU(0.2, True)

        # Pooling for horizontal, vertical, and diagonal directions
        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_diag = nn.AdaptiveAvgPool2d((None, None))

        # Convolutions for each direction
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0))
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1))
        self.conv_diag = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 3), padding=1,
                                   dilation=2)  # Diagonal (rotated kernel)

        self.fuse_conv = nn.Conv2d(midplanes * 3, midplanes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.conv_final = nn.Conv2d(midplanes, outplanes, kernel_size=1)

        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)

    def forward(self, x):
        _, _, h, w = x.size()

        x = self.conv_4_1(x)
        x = self.relu_4_1(x)

        # Horizontal direction processing
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)

        # Vertical direction processing
        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)

        # Diagonal direction processing
        x_diag = self.pool_diag(x)
        x_diag = self.conv_diag(x_diag)
        x_diag = F.interpolate(x_diag, size=(h, w), mode='bilinear', align_corners=False)

        x_all = torch.cat([x_1_h, x_1_w, x_diag], dim=1)
        hx = self.relu(self.fuse_conv(x_all))

        mask_1 = self.conv_final(hx).sigmoid()
        out1 = x * mask_1

        return out1

class GDAA(nn.Module):

    def __init__(self, in_size, out_size):
        super(GDAA, self).__init__()

        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)

        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)

        out_1, out_2 = torch.chunk(out, 2, dim=1)

        out = torch.cat([self.norm(out_1), out_2], dim=1)

        out = self.relu_1(out)

        out = self.relu_2(self.conv_2(out))

        out = out + x

        return out

class MCAM(nn.Module):
    def __init__(self, in_size, out_size):
        super(MCAM, self).__init__()

        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.LeakyReLU(0.2, True)
        self.conv_2 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.relu_2 = nn.LeakyReLU(0.2, True)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu_3 = nn.LeakyReLU(0.2, True)

        self.conv_4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.relu_4 = nn.LeakyReLU(0.2, True)

        #self.conv_4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #self.relu_4_1 = nn.LeakyReLU(0.2, True)
        self.conv_4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu_4_2 = nn.LeakyReLU(0.2, True)
        self.conv_4_3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.relu_4_3 = nn.LeakyReLU(0.2, True)

        self.GSA_Block = G_Block(64, 64)
        self.LPA_Block = L_Block()

        self.GDA_Block = GDAA(in_size, out_size)

    def forward(self, x):
        hx1 = self.conv_1(x)
        hx1 = self.relu_1(hx1)

        hx1 = self.conv_4(hx1)
        hx1 = self.relu_4(hx1)

        hx31 = self.conv_4_3(hx1)
        hx31 = self.relu_4_3(hx31)

        GSA1 = self.GSA_Block(hx1)
        LPA1 = self.LPA_Block(hx1)

        # Ensure all outputs have the same channel dimensions
        GSA1 = self._adjust_channels_for_concat(GSA1, 64)
        LPA1 = self._adjust_channels_for_concat(LPA1, 64)
        hx31 = self._adjust_channels_for_concat(hx31, 64)

        hx1 = torch.cat([GSA1, LPA1, hx31], dim=1)

        hx1 = self.conv_2(hx1)
        hx1 = self.relu_2(hx1)

        hx1 = self.conv_3(hx1)
        hx1 = self.relu_3(hx1)

        hx1 = hx1 + x
        hx1 = self.GDA_Block(hx1) + hx1

        hx = hx1 + self.identity(x)

        return hx


    def _adjust_channels_for_concat(self, x, target_channels):
        """ Helper function to adjust the channels of tensors before concatenation """
        _, _, h, w = x.size()
        current_channels = x.size(1)
        if current_channels != target_channels:
            # Apply a 1x1 conv to adjust the channels
            conv = nn.Conv2d(current_channels, target_channels, kernel_size=1)
            x = conv(x)
        return x


class ARB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size):
        super(ARB, self).__init__()

        self.dgm1 = ARM(n_feat, n_feat, kernel_size)
        self.dgm2 = ARM(n_feat, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''

        out = self.relu(self.dgm1(x, inter))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dgm2(out, inter))
        out = self.conv2(out) + x

        return out

class ARG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks):
        super(ARG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            ARB(conv, n_feat, kernel_size) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        return res

class RT(nn.Module):
    def __init__(self, opt, conv=default_conv):
        super(RT, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        modules_head.append(MCAM(n_feats, n_feats))
        self.head = nn.Sequential(*modules_head)

        # body ARN
        modules_body = [
            ARG(default_conv, n_feats, kernel_size, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)


    def forward(self, x, inter):
        # head
        x = self.head(x)

        # body ARN
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x
