from model import Encoder, Generator, NLayerDiscriminator, \
    SemanticDiscriminator, Normal_Discriminator, SegNet, ResDecoder, ResEncoder, \
    Encoder_2, Decoder_2, Muti_Discriminator
from MyFun import cal_grad_penalty, quantizer
from loader import Cityscapes, Cityscapes_2w
from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
from SSIM import SSIM
from Slice_model import Slice_Generator, Channel_Slice_decoder, Channel_Slice_encoder, Fading_Slice_encoder, Fading_Slice_decoder
import Communication_module

ssim_fn = SSIM()

batch_size = 3
semantic_feature_num = 256
load_flag = 0  # 1:load init 2:load next 0:not load

dataset = Cityscapes(datasetpath="../../Dataset/Cityscapes/dataset-train-label.txt", labelsetpath="../../Dataset/Cityscapes/gtFine/")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_dataset = Cityscapes(datasetpath="../../Dataset/Cityscapes/dataset-val-label.txt", labelsetpath="../../Dataset/Cityscapes/gtFine/", kind="val")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

"----------------------Base model------------------------"
C_list = [4, 8, 16, 32, 64, 128]
C = C_list[0]
print("semantic  bpp:%f, cr:%f" % (24 / (1024 * 512 * 3 * 8 / (C * 30 * 62 * np.log2(7))), 1024 * 512 * 3 * 8 / (30 * 62 * C * np.log2(7))))
print("enhance bpp:%f, cr:%f" % (24 / (1024 * 512 * 3 * 8 / (32 * 32 * 64 * np.log2(7))), (1024 * 512 * 3 * 8 / (32 * 32 * 64 * np.log2(7)))))
Semantic_encoder = Encoder_2(C).cuda()
Semantic_decoder = Decoder_2(C).cuda()

"----------------------Enhance model------------------------"
down_index = 4
downsample_size = np.array([[256, 512],
                            [128, 256],
                            [64, 128],
                            [32, 64],
                            [192, 384],
                            ])

x_size = downsample_size[down_index][0]
y_size = downsample_size[down_index][1]


averagepool = nn.AdaptiveAvgPool2d((x_size, y_size))

Slice_decoder = Slice_Generator().cuda()

Encoder_load = "./model/Encoder_2_q_" + str(C) + ".pkl"
Decoder_load = "./model/Decoder_2_q_" + str(C) + ".pkl"

Encoder_save = "./model/Encoder_2_q_" + str(C) + ".pkl"
Decoder_save = "./model/Decoder_2_q_" + str(C) + ".pkl"


Slice_decoder_save = "./model/Slice_decoder_" + str(down_index) + ".pkl"
Semantic_encoder.load_state_dict(torch.load(Encoder_load))
Semantic_decoder.load_state_dict(torch.load(Decoder_load))
Slice_decoder.load_state_dict(torch.load(Slice_decoder_save))

"----------------------Channel model------------------------"
Channel_encoder = Fading_Slice_encoder(C).cuda()
Channel_decoder = Fading_Slice_decoder(C).cuda()
Channel_encoder_save = "./model/Channel_encoder_fading_10dB_" + str(C) + ".pkl"
Channel_decoder_save = "./model/Channel_decoder_fading_10dB_" + str(C) + ".pkl"


#Channel_encoder.load_state_dict(torch.load(Channel_encoder_save))
#Channel_decoder.load_state_dict(torch.load(Channel_decoder_save))

opt = torch.optim.Adam([
    {'params': Channel_encoder.parameters(), 'lr': 0.0001},
     {'params': Channel_decoder.parameters(), 'lr': 0.0001}])



num_epoch = 100000
mse_fn = nn.MSELoss()
bce_loss = nn.BCELoss()

channel = "fading"
for epoch in range(num_epoch):
    semantic_coding_q = torch.randint(-3, 4, (3, 4, 32, 64)).type(torch.FloatTensor).cuda()
    "------------------------Channel Slice model----------------"
    if channel == "fading":
        semantic_coding_q_pass = Communication_module.Fading_channel_pass(10, semantic_coding_q)
        semantic_coding_q_half = semantic_coding_q_pass[:, :C // 2, :, :]

        semantic_coding_q_half_pass = Communication_module.Fading_channel_pass(10, semantic_coding_q_half)
        channel_info = torch.cat((semantic_coding_q, semantic_coding_q_half_pass), dim=1)

        channel_info_coding = Channel_encoder(channel_info)
        channel_info_coding = Communication_module.Fading_channel_pass(10, channel_info_coding)
        channel_info_decoding = Channel_decoder(channel_info_coding)
    else:
        semantic_coding_q_pass = Communication_module.AWGN(10, semantic_coding_q)
        semantic_coding_q_half = semantic_coding_q_pass[:, :C // 2, :, :]

        semantic_coding_q_pass_2 = Communication_module.AWGN(10, semantic_coding_q_half)
        channel_info = torch.cat((semantic_coding_q, semantic_coding_q_pass_2), dim=1)

        channel_info_coding = Channel_encoder(channel_info)
        channel_info_coding = Communication_module.AWGN(10, channel_info_coding)
        channel_info_decoding = Channel_decoder(channel_info_coding)


    ori_loss = mse_fn(semantic_coding_q_pass, semantic_coding_q)
    mse_loss = mse_fn(channel_info_decoding, semantic_coding_q)


    opt.zero_grad()
    mse_loss.backward()
    opt.step()

    "val"

    if epoch%100==0:

        torch.save(Channel_encoder.state_dict(), Channel_encoder_save)
        torch.save(Channel_decoder.state_dict(), Channel_decoder_save)

        with torch.no_grad():
            for i_batch, data_batch in enumerate(dataloader):
                images, nature, flat, human_vehicle, label = data_batch

                images = images.type(torch.FloatTensor).cuda()

                "---------------------Base model---------------------"
                semantic_coding = Semantic_encoder(images)  # 512*8*16

                semantic_coding = (semantic_coding) * 3

                semantic_coding_q = quantizer(semantic_coding, L=7)
                "--------------------Enhance model-------------------"
                recon_images = Semantic_decoder(semantic_coding_q)

                enhance_images = (images - recon_images)  # *human_vehicle
                max_h = torch.max(enhance_images)
                min_h = torch.min(enhance_images)
                enhance_images = (enhance_images - min_h) / (max_h - min_h)
                enhance_images_avg = averagepool(enhance_images)
                enhance_recon = Slice_decoder(enhance_images_avg)
                enhance_recon = enhance_recon * (max_h - min_h) + min_h

                if channel == "fading":
                    semantic_coding_q_pass = Communication_module.Fading_channel_pass(10, semantic_coding_q)
                    semantic_coding_q_half = semantic_coding_q_pass[:, :C // 2, :, :]

                    semantic_coding_q_half_pass = Communication_module.Fading_channel_pass(10, semantic_coding_q_half)
                    channel_info = torch.cat((semantic_coding_q, semantic_coding_q_half_pass), dim=1)

                    channel_info_coding = Channel_encoder(channel_info)
                    #channel_info_coding = Communication_module.Fading_channel_pass(10, channel_info_coding)
                    channel_info_decoding = Channel_decoder(channel_info_coding)

                else:
                    semantic_coding_q_pass = Communication_module.AWGN(10, semantic_coding_q)
                    semantic_coding_q_half = semantic_coding_q_pass[:, :C // 2, :, :]

                    semantic_coding_q_half_pass = Communication_module.AWGN(10, semantic_coding_q_half)
                    channel_info = torch.cat((semantic_coding_q, semantic_coding_q_half_pass), dim=1)

                    channel_info_coding = Channel_encoder(channel_info)
                    channel_info_coding = Communication_module.AWGN(10, channel_info_coding)
                    channel_info_decoding = Channel_decoder(channel_info_coding)

                "-------------------Recover image-----------------------"
                recon_images_channel = Semantic_decoder(channel_info_decoding)
                recon_images_pass = Semantic_decoder(semantic_coding_q_pass)


                enhance_recon_image_channel = enhance_recon + recon_images_channel
                enhance_recon_image_pass = enhance_recon + recon_images_pass
                enhance_image = enhance_recon+recon_images


                images = images.cpu().detach().numpy()
                enhance_recon_image_pass = enhance_recon_image_pass.cpu().detach().numpy()
                enhance_recon_image_channel = enhance_recon_image_channel.cpu().detach().numpy()
                enhance_image= enhance_image.cpu().detach().numpy()

                images = (images + 1) / 2
                enhance_recon_image_pass = (enhance_recon_image_pass + 1) / 2
                enhance_recon_image_channel = (enhance_recon_image_channel + 1) / 2
                enhance_image = (enhance_image + 1) / 2

                real_ori_loss = mse_fn(semantic_coding_q_pass, semantic_coding_q)
                real_mse_loss = mse_fn(channel_info_decoding, semantic_coding_q)
                print("[E: %d/%d] , mse_loss: %f, ori_loss:%f, real mse_loss: %f, real ori_loss:%f, channel_mse:%f, , ori_mse:%f, no_pass_mse:%f"
                      % (epoch, num_epoch, mse_loss, ori_loss, real_mse_loss, real_ori_loss,
                         ((images - enhance_recon_image_channel) ** 2).mean(),
                         ((images - enhance_recon_image_pass) ** 2).mean(),
                         ((images - enhance_image) ** 2).mean()
                         ))

                break
                '''
                plt.figure(1)
                plt.subplot(311)
                plt.imshow(np.transpose(images[0], (1, 2, 0)))
                plt.subplot(312)
                plt.imshow(np.transpose(recon_images[0], (1, 2, 0)))
                plt.subplot(313)
                plt.imshow(np.transpose(enhance_recon_image[0], (1, 2, 0)))
                plt.show()
                '''



