from model import Encoder, Generator, NLayerDiscriminator, \
    SemanticDiscriminator, Normal_Discriminator, SegNet, ResDecoder, ResEncoder, \
    Encoder_2, Decoder_2, Muti_Discriminator
from MyFun import cal_grad_penalty, quantizer
from loader import Cityscapes, Cityscapes_2w, OpenImage, Loader_all_data, read_all_image
from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
from SSIM import SSIM
import Communication_module


if __name__ == '__main__':
    ssim_fn = SSIM()

    batch_size = 8
    semantic_feature_num = 256
    load_flag = 1 # 1:load init 2:load next 0:not load

    '''
    dataset = Cityscapes_2w(datasetpath="../../Dataset/Cityscapes/dataset-train.txt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = Cityscapes_2w(datasetpath="../../Dataset/Cityscapes/dataset-val.txt")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    dataset = Cityscapes(datasetpath="../../Dataset/Cityscapes/dataset-train-label.txt", labelsetpath="../../Dataset/Cityscapes/gtFine/")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = Cityscapes(datasetpath="../../Dataset/Cityscapes/dataset-val-label.txt", labelsetpath="../../Dataset/Cityscapes/gtFine/", kind="val")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    '''
    '''
    dataset = OpenImage('../dataset/OpenImage/train_0.txt')
    dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(len(dataset))
    '''


    val_dataset = OpenImage('../dataset/OpenImage/validation.txt')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    '''
    val_data = read_all_image('../dataset/OpenImage/validation.txt', 200)
    val_dataset = Loader_all_data(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    '''

    C_list = [4, 8, 16, 32, 64, 128]

    C = C_list[0]
    print("semantic  bpp:%f, cr:%f" % (24/(512*512*24/(C*30*30*np.log2(7))), 512*512*24/(30*30*C*np.log2(7))))

    Semantic_encoder = Encoder_2(C).cuda()
    Semantic_decoder = Decoder_2(C).cuda()
    Muti_discriminator = Muti_Discriminator().cuda()

    # Semantic_discriminator = SemanticDiscriminator(semantic_feature_num).cuda()

    Encoder_load = "./model/OpenImage_Encoder_q_" + str(C) + ".pkl"
    Decoder_load = "./model/OpenImage_Decoder_q_" + str(C) + ".pkl"

    Encoder_save = "./model/OpenImage_Encoder_q_" + str(C) + "_gan.pkl"
    Decoder_save = "./model/OpenImage_Decoder_q_" + str(C) + "_gan.pkl"

    Semantic_discriminator_save = "./model/Semantic_discriminator_res.pkl"
    Muti_discriminator_save = "./model/Muti_discriminator_res.pkl"
    Muti_discriminator.load_state_dict(torch.load(Muti_discriminator_save))


    if load_flag == 1:
        Semantic_encoder.load_state_dict(torch.load(Encoder_save))
        Semantic_decoder.load_state_dict(torch.load(Decoder_save))
        # Semantic_discriminator.load_state_dict(torch.load(Semantic_discriminator_save))
        # Patch_discriminator.load_state_dict(torch.load(Patch_discriminator_save))


    opt = torch.optim.Adam(
        [{'params': Semantic_encoder.parameters(), 'lr': 0.0001},
         {'params': Semantic_decoder.parameters(), 'lr': 0.0001}])
    MD_optimizer = torch.optim.Adam(Muti_discriminator.parameters(), lr=0.0001)


    # SD_optimizer = torch.optim.RMSprop(Semantic_discriminator.parameters(), lr=0.0002)

    loss_list = []
    g_loss_list = []
    d_loss_list = []
    perce_loss_list = []
    num_epoch = 250
    mse_fn = nn.MSELoss()
    semantic_weight = 0.0001
    g_weight = 0.001
    lambda_gp = 0.1
    p_weight = 0.01

    bce_loss = nn.BCELoss()

    for epoch in range(num_epoch):
        for i_batch, data_batch in enumerate(dataloader):
            if i_batch>0:
                print("data:"+str(time2-time8))
            images = data_batch

            images = torch.permute(images, [0, 3, 2, 1])

            images = images.type(torch.FloatTensor).cuda()

            semantic_coding = Semantic_encoder(images)  # 512*8*16

            semantic_coding = (semantic_coding)*3
            semantic_coding_q = quantizer(semantic_coding, L=7)

            recon_images = Semantic_decoder(semantic_coding_q)

            "mse loss"
            images = (images + 1) / 2
            recon_images = (recon_images + 1) / 2

            L2 = ((recon_images - images) ** 2).mean()
            L1 = torch.abs(recon_images-images).mean()
            ssim_loss = 1 - ssim_fn(images, recon_images)

            "patch loss 图像真实性判别器 判别器需要识别重建图像和原始图像，生成器与之对抗，尽量缩小重建图像与原始图像的差别，不让判别器识别"

            real_output = Muti_discriminator(images)

            fake_output_d = Muti_discriminator(recon_images.detach())
            fake_output_g = Muti_discriminator(recon_images)
            '''
            valid_label = Variable(torch.cuda.FloatTensor(images.size(0), 1).fill_(0.9))
            fake_label = Variable(torch.cuda.FloatTensor(images.size(0), 1).fill_(0.1))
    
            g_loss = bce_loss(fake_output_g, valid_label)
            d_loss = bce_loss(real_output, valid_label) + bce_loss(fake_output_d, fake_label)
            '''
            #print("Dreal:%f, Dfake:%f"%(torch.mean(real_output), torch.mean(fake_output_g)))
            
            "WGAN-GP loss"
            g_loss = -torch.mean(fake_output_g)

            grad_penalty = cal_grad_penalty(Muti_discriminator, images.data, recon_images.data)

            d_loss = -torch.mean(real_output) + torch.mean(fake_output_d) + lambda_gp * grad_penalty  # 改进2、生成器和判别器的loss不取log


            "Update"
            ae_loss = L2 + 0.01*ssim_loss+ 0.0001*g_loss
            opt.zero_grad()
            ae_loss.backward()
            opt.step()


            "train patch D"
            md_loss = d_loss * 0.1
            MD_optimizer.zero_grad()
            md_loss.backward()
            MD_optimizer.step()

            perce_loss = 0
            #d_loss = 0
            #g_loss = 0


            print("[E: %d/%d] [L:%d/%d], L1:%f, L2: %f, ssim:%f, p_loss:%f, d_loss:%f, g_loss:%f" % (
            epoch, num_epoch, i_batch, len(val_dataloader), L1, L2.data, ssim_loss.data, perce_loss, d_loss, g_loss))
            print("")
            "mse_loss: 0.001 0.0009"


            #perce_loss_list.append(perce_loss.data.cpu())

            if i_batch % 50 == 0:
                loss_list.append(L2.data.cpu())
                #g_loss_list.append(g_loss.data.cpu())
                #d_loss_list.append(d_loss.data.cpu())
                torch.save(Semantic_encoder.state_dict(), Encoder_save)
                torch.save(Semantic_decoder.state_dict(), Decoder_save)
                # torch.save(Semantic_discriminator.state_dict(), Semantic_discriminator_save)
                torch.save(Muti_discriminator.state_dict(), Muti_discriminator_save)
                '''
                images = images.cpu().detach().numpy()
    
                recon_images = recon_images.cpu().detach().numpy()
            
                plt.figure(1)
                plt.subplot(211)
                plt.imshow(np.transpose(images[0], (1, 2, 0)))
                plt.subplot(212)
                plt.imshow(np.transpose(recon_images[0], (1, 2, 0)))
                plt.show()
                
                plt.figure(2)
                plt.subplot(311)
                plt.plot(loss_list)
                plt.title("mse")
                plt.subplot(312)
                plt.plot(perce_loss_list)
                plt.title("perce_loss")
                plt.subplot(313)
                plt.plot(g_loss_list)
                plt.plot(d_loss_list) 
                plt.legend(["g_loss", "d_loss"])
                plt.show()

                for i_batch, data_batch in enumerate(val_dataloader):
                    images = data_batch
    
                    images = images.type(torch.FloatTensor).cuda()
    
                    semantic_coding = Semantic_encoder(images)  # 512*4*2
    
                    semantic_coding = semantic_coding * 3
    
                    semantic_coding_q = quantizer(semantic_coding, L=7)
    
                    recon_images = Semantic_decoder(semantic_coding_q)
                    "mse loss"
                    images = (images + 1) / 2
                    recon_images = (recon_images + 1) / 2
                    mse_loss = ((recon_images - images) ** 2).mean()
                    ssim = ssim_fn(recon_images, images)
                    print(mse_loss, ssim)
                    break
                '''
