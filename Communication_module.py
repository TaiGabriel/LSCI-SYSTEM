import torch
import numpy as np
import torch.nn as nn
from scipy import sparse

device = torch.device("cuda:0")
def AWGN(sinr, coding):
    device = coding.device
    "ADD Noise"
    coding_shape = coding.shape  #batch_size * (z_dim*2) * 2
    coding_reshape = coding.reshape(coding_shape[0], -1)

    # normalize latent vector so that the average power is 1
    #coding_reshape = np.sqrt(coding_reshape.shape[1]) * nn.functional.normalize(coding_reshape, p=2, dim=1)

    power = torch.sum(coding_reshape * torch.conj(coding_reshape)) / (coding_reshape.shape[0] * coding_reshape.shape[1])
    noise_stddev = np.sqrt(10 ** (-sinr / 10))*power

    n = torch.randn(coding_reshape.shape).to(device)

    coding_noise = coding_reshape + n * noise_stddev

    coding_noise = coding_noise.reshape(coding_shape)
    return coding_noise

def Fading_Channel(input_com, sinr):
    [batch_size, length] = input_com.shape
    coding_shape = input_com.shape  # batch_size * (z_dim*2) * 2

    # normalize the latent vector so that the average power is 1
    #z_norm = torch.sum(input_com * torch.conj(input_com))
    #input_com = input_com * torch.sqrt(length / z_norm)

    beta = 1 #信道增益 大尺度损耗
    h = beta * torch.complex(torch.randn(coding_shape)*(1 / np.sqrt(2)), torch.randn(coding_shape)*(1 / np.sqrt(2))).to(device)

    y_h = h * input_com

    power = torch.sum(input_com * torch.conj(input_com)) / (batch_size*length)

    noise_stddev = np.sqrt(10 ** (-sinr / 10))*power

    awgn = torch.complex(
        torch.randn(coding_shape)*(1 / np.sqrt(2)),
        torch.randn(coding_shape)*(1 / np.sqrt(2))
    ).to(device)

    y_add = y_h + awgn * noise_stddev
    y_add = y_add/h
    return y_add, h

def Fading_channel_pass(sinr, input):
    or_shape = input.shape

    input_reshape = input.reshape(input.shape[0], -1, 2)
    input_com = torch.complex(input_reshape[:, :, 0], input_reshape[:, :, 1]).to(device)

    "Fading Channel"
    [input_add, perfect_CSI] = Fading_Channel(input_com, sinr)

    output = torch.zeros(input_reshape.shape).to(input.device)
    output[:, :, 0] = torch.real(input_add)
    output[:, :, 1] = torch.imag(input_add)

    output = output.reshape(or_shape)

    return output


def Constellation_mapping(data):
    C = np.cos(3.14/8)
    S = np.sin(3.14/8)
    I = [C, S, -S, -C, -C, -S, S, C]
    Q = [S, C, C, S, -S, -C, -C, -S]

    string = ["000", "001", "010", "011",
              "100", "101", "110", "111"]
    index = string.index(data)

    return I[index], Q[index]


def Quantization(data, num_bits):
    "Quantization"
    data_quan = data * (2**num_bits - 1)
    data_quan = torch.tensor(data_quan, dtype = torch.int)

    return data_quan

def QAM_modulation(data_quan, num_bit):
    batch_size = data_quan.shape[0]
    z_dims = data_quan.shape[1]
    "QAM"
    binary_repr_v = np.vectorize(np.binary_repr)

    data_bin = binary_repr_v(data_quan, num_bit)
    group_num = num_bit // 3

    IQ = np.zeros((batch_size, z_dims * group_num, 2))
    for m in range(batch_size):
        k = 0
        for n in range(z_dims):
            for i in range(group_num):
                [I, Q] = Constellation_mapping(data_bin[m][n][3 * i:3 * (i + 1)])
                IQ[m][k][0] = I
                IQ[m][k][1] = Q

                k = k + 1

    IQ = torch.tensor(IQ).to(device)

    return IQ

def QAM_demodulation(IQ, num_bit):
    group_num = num_bit // 3
    batch_size = IQ.shape[0]
    z_dims = IQ.shape[1] // group_num
    "Demodulation"
    C = np.cos(3.14 / 8)
    S = np.sin(3.14 / 8)

    IQ_MAP = torch.tensor([[C, S, -S, -C, -C, -S, S, C],
                           [S, C, C, S, -S, -C, -C, -S]], device=device)
    IQ_MAP = IQ_MAP.view(1, 1, 2, 8)
    IQ_MAP_reap = IQ_MAP.repeat(IQ.shape[0], IQ.shape[1], 1, 1)

    coding_noise = IQ.view(IQ.shape[0], IQ.shape[1], 2, 1)
    coding_noise_reap = coding_noise.repeat(1, 1, 1, 8)

    distance = torch.abs((coding_noise_reap[:, :, 0, :] - IQ_MAP_reap[:, :, 0, :])) + torch.abs(
        (coding_noise_reap[:, :, 1, :] - IQ_MAP_reap[:, :, 1, :]))
    min_distance = torch.argmin(distance, 2).cpu()
    string = np.array(["000", "001", "010", "011",
              "100", "101", "110", "111"])

    recover_data = torch.zeros(batch_size, z_dims)
    for i in range(batch_size):
        k = 0
        for j in range(z_dims):
            string1 = ""
            for m in range(group_num):
                string1 += string[min_distance[i][k * group_num + m]]
            k = k + 1
            recover_data[i, j] = int(string1, 2)

    return recover_data

def modulation(data, num_bit):
    max_ = torch.max(data)
    min_ = torch.min(data)
    coding = (data - min_) / (max_ - min_)
    coding_quan = Quantization(coding, num_bit).cpu().detach().numpy()
    coding_qam = QAM_modulation(coding_quan, num_bit)

    coding_dqam = QAM_demodulation(coding_qam, num_bit)
    coding_dqam = (coding_dqam / (2**num_bit)) * (max_ - min_) + min_

    return coding_dqam

def communication_process(data, num_bit, ser):
    #device = data.device
    #data = torch.tensor(data + 3, dtype=torch.int)

    #data = data.cpu().detach().numpy()
    binary_repr_v = np.vectorize(np.binary_repr)

    data_bin = binary_repr_v(data, num_bit)

    length_bit = data_bin.shape[0] * data_bin.shape[1] * num_bit

    num_error_bit = int(length_bit * ser)

    error_index = np.random.randint(0, length_bit, (1, num_error_bit))

    x = error_index//(data_bin.shape[1] * num_bit)
    y = error_index % (data_bin.shape[1] * num_bit)
    y2 = y // num_bit
    y3 = y % num_bit
    for i in range(num_error_bit):
        data_string = data_bin[x[0, i]][y2[0, i]]
        t = list(data_string)
        if data_string[y3[0, i]] == "0":
            t[y3[0, i]] = '1'
        else:
            t[y3[0, i]] = '0'
        data_bin[x[0, i]][y2[0, i]] = ''.join(t)

    recover_data = torch.zeros(data.shape)
    for i in range(recover_data.shape[0]):
        for j in range(recover_data.shape[1]):
            recover_data[i, j] = int(data_bin[i, j], 2)

    #recover_data = recover_data.to(device)
    #recover_data = recover_data - 3

    return recover_data