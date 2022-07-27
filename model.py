import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torch.nn.functional as F

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


class BasicBlock(tnn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tnn.Conv2d(inplanes, planes, stride)
        self.bn1 = tnn.InstanceNorm2d(planes)
        self.relu = tnn.ReLU(inplace=True)
        self.conv2 = tnn.Conv2d(inplanes, planes, stride)
        self.bn2 = tnn.InstanceNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out += identity
        out = self.relu(out)

        return out


class Encoder(tnn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = tnn.Sequential(
            tnn.Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )


        self.layer2 = tnn.Sequential(
            tnn.Conv2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )

        self.layer3 = tnn.Sequential(
            tnn.Conv2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )

        self.layer4 = tnn.Sequential(
            tnn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )

        self.layer5 = tnn.Sequential(
            tnn.Conv2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.Sigmoid()
        )


    def forward(self, x):
        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.layer5(out)

        return out

class Generator(tnn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.deconv_1 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )
        self.deconv_2 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )

        self.deconv_3 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )
        self.deconv_4 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )

        self.deconv_5 = tnn.Sequential(
            tnn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.Tanh()
        )


    def forward(self, x):

        out = self.deconv_1(x)

        out = self.deconv_2(out)

        out = self.deconv_3(out)

        out = self.deconv_4(out)

        out = self.deconv_5(out)

        return out

class NLayerDiscriminator(tnn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=tnn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()


        kw = 4
        padw = 1
        sequence = [tnn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), tnn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                tnn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult, affine=True),
                tnn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            tnn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult, affine=True),
            tnn.LeakyReLU(0.2, True)
        ]

        sequence += [tnn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = tnn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        input = self.model(input)
        #input = F.sigmoid(input)
        return input

class SemanticDiscriminator(tnn.Module):
    def __init__(self, input_channels):
        super(SemanticDiscriminator, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.deconv_1 = tnn.Sequential(
            tnn.ConvTranspose2d(input_channels, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.BatchNorm2d(512),
            tnn.ReLU()
        )
        self.deconv_2 = tnn.Sequential(
            tnn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.BatchNorm2d(256),
            tnn.ReLU()
        )

        self.deconv_3 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.BatchNorm2d(128),
            tnn.ReLU()
        )
        self.deconv_4 = tnn.Sequential(
            tnn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.BatchNorm2d(64),
            tnn.ReLU()
        )

        self.deconv_5 = tnn.Sequential(
            tnn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.BatchNorm2d(32),
            tnn.ReLU()
        )

        self.deconv_6 = tnn.Sequential(
            tnn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.BatchNorm2d(16),
            tnn.ReLU()
        )

        self.deconv_7 = tnn.Sequential(
            tnn.ConvTranspose2d(16, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.Tanh()
        )

    def forward(self, x):
        out = self.deconv_1(x)
        out = self.deconv_2(out)
        out = self.deconv_3(out)

        out = self.deconv_4(out)
        out = self.deconv_5(out)
        out = self.deconv_6(out)
        out = self.deconv_7(out)

        return out


class Normal_Discriminator(tnn.Module):
    def __init__(self, channels_img=3, features_d=32):
        super(Normal_Discriminator, self).__init__()
        self.disc = tnn.Sequential(
            # input: N x channels_img x 64 x 64
            tnn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            tnn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            tnn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

        self.linear = tnn.Linear(15*31, 1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return tnn.Sequential(
            tnn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            tnn.InstanceNorm2d(out_channels, affine=True),
            tnn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.disc(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x



class SegNet(tnn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = tnn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = tnn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = tnn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = tnn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = tnn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = tnn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv31d = tnn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = tnn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv21d = tnn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = tnn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv11d = tnn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x1p, id1 = F.max_pool2d(x11,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x2p, id2 = F.max_pool2d(x21,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x3p, id3 = F.max_pool2d(x31,kernel_size=2, stride=2,return_indices=True)

        # Stage 3d
        x3d = F.max_unpool2d(x3p, id3, kernel_size=2, stride=2)
        x31d = F.relu(self.bn31d(self.conv31d(x3d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x21d = F.relu(self.bn21d(self.conv21d(x2d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)

        x11d = self.conv11d(x1d)

        return x1p, x2p, x3p, x11d



class ResEncoder(tnn.Module):
    def __init__(self, ):
        super(ResEncoder, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = tnn.Sequential(
            tnn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(64),
            tnn.LeakyReLU()
        )

        self.layer2 = tnn.Sequential(
            tnn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )

        self.layer3 = tnn.Sequential(
            tnn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )

        self.layer4 = tnn.Sequential(
            tnn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.LeakyReLU()
        )

        self.layer5 = tnn.Sequential(
            tnn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.LeakyReLU()
        )

        self.layer6 = tnn.Sequential(
            tnn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.Sigmoid()
            #tnn.LeakyReLU()
        )



    def forward(self, x):
        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.layer5(out)

        out = self.layer6(out)

        return out

class ResDecoder(tnn.Module):
    def __init__(self,):
        super(ResDecoder, self).__init__()
        self.deconv_1 = BasicBlock(512, 512)
        self.deconv_2 = BasicBlock(512, 512)

        self.upsample_1 = tnn.Sequential(
            tnn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.ReLU()
        )

        self.deconv_3 = BasicBlock(512, 512)

        self.upsample_2 = tnn.Sequential(
            tnn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.ReLU()
        )

        self.deconv_4 = BasicBlock(256, 256)

        self.upsample_3 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.ReLU()
        )

        self.deconv_5 = BasicBlock(128, 128)

        self.upsample_4 = tnn.Sequential(
            tnn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(64),
            tnn.ReLU()
        )

        self.deconv_6 = BasicBlock(64, 64)

        self.upsample_5 = tnn.Sequential(
            tnn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(32),
            tnn.ReLU()
        )

        self.deconv_7 = BasicBlock(32, 32)

        self.upsample_6 = tnn.Sequential(
            tnn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.Tanh()
        )



    def forward(self, x):
        out = self.deconv_1(x)
        out = self.deconv_2(out)
        out = self.upsample_1(out)

        out = self.deconv_3(out)

        out = self.upsample_2(out)
        out = self.deconv_4(out)

        out = self.upsample_3(out)
        out = self.deconv_5(out)

        out = self.upsample_4(out)
        out = self.deconv_6(out)

        out = self.upsample_5(out)
        out = self.deconv_7(out)

        out = self.upsample_6(out)

        return out

class Encoder_2(tnn.Module):
    def __init__(self, C):
        super(Encoder_2, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = tnn.Sequential(
            tnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(64),
            tnn.LeakyReLU()
        )

        self.layer2 = tnn.Sequential(
            tnn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )

        self.layer3 = tnn.Sequential(
            tnn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )

        self.layer4 = tnn.Sequential(
            tnn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.LeakyReLU()
        )

        self.layer5 = tnn.Sequential(
            tnn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(1024),
            tnn.LeakyReLU()
        )

        self.layer6 = tnn.Sequential(
            tnn.Conv2d(1024, C, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(C),
            tnn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x


class Decoder_2(tnn.Module):
    def __init__(self, C):
        super(Decoder_2, self).__init__()

        self.upsample_1 = tnn.Sequential(
            tnn.Conv2d(C, 1024, kernel_size=(3, 3), stride=(1, 1)),
            tnn.InstanceNorm2d(1024),
            tnn.LeakyReLU()
        )
        self.resblock_1 = BasicBlock(1024, 1024)
        self.resblock_2 = BasicBlock(1024, 1024)
        self.resblock_3 = BasicBlock(1024, 1024)
        self.resblock_4 = BasicBlock(1024, 1024)
        self.resblock_5 = BasicBlock(1024, 1024)
        self.resblock_6 = BasicBlock(1024, 1024)
        self.resblock_7 = BasicBlock(1024, 1024)
        self.resblock_8 = BasicBlock(1024, 1024)
        self.resblock_9 = BasicBlock(1024, 1024)

        self.upsample_2 = tnn.Sequential(
            tnn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.ReLU()
        )
        self.upsample_3 = tnn.Sequential(
            tnn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.ReLU()
        )
        self.upsample_4 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.ReLU()
        )

        self.upsample_5 = tnn.Sequential(
            tnn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1)),
            tnn.InstanceNorm2d(64),
            tnn.ReLU()
        )

        self.upsample_6 = tnn.Sequential(
            tnn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1)),
            tnn.Tanh()
        )

        self.pad = tnn.ReflectionPad2d(1)
    def forward(self, x):
        out = self.upsample_1(x)

        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = self.resblock_5(out)
        out = self.resblock_6(out)
        out = self.resblock_7(out)
        out = self.resblock_8(out)
        out = self.resblock_9(out)

        out = self.upsample_2(out)
        out = self.upsample_3(out)
        out = self.upsample_4(out)
        out = self.pad(out)
        out = self.upsample_5(out)

        out = self.upsample_6(out)

        return out


class Muti_Discriminator(tnn.Module):
    def __init__(self, ):
        super(Muti_Discriminator, self).__init__()

        self.D1 = tnn.Sequential(
            tnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            tnn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),
            tnn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            tnn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1)),
            tnn.Sigmoid()
        )

        self.D2 = tnn.Sequential(
            tnn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2)),
            tnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            tnn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),
            tnn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            tnn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1)),
            tnn.Sigmoid()
        )

        self.D3 = tnn.Sequential(
            tnn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2)),
            tnn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2)),
            tnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            tnn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),
            tnn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            tnn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1)),
            tnn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.D1(x)
        y2 = self.D2(x)
        y3 = self.D3(x)

        y1 = y1.reshape(y1.shape[0], -1)
        y2 = y2.reshape(y2.shape[0], -1)
        y3 = y3.reshape(y3.shape[0], -1)

        y = (torch.mean(y1, 1) + torch.mean(y2, 1) + torch.mean(y3, 1))/3
        y = y.reshape(-1, 1)
        return y




'''
encoder = Encoder_2(C = 128).cuda()
decoder = Decoder_2(C = 128).cuda()

x = torch.rand(6, 3, 1024, 512).cuda()
encoding = encoder(x)

y = decoder(encoding)

print(encoding.shape, y.shape)
'''