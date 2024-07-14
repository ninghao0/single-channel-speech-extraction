import numpy as np
import librosa
import os
import torch
# import torchaudio.transforms
from pesq import pesq
from pystoi import stoi


BASE1_DIR = "SpEx1/SNR=0/spex/"
BASE2_DIR = "data/tt-finalplusplus/clean/"
filenames1 = os.listdir(BASE1_DIR)
filenames2 = os.listdir(BASE2_DIR)



path1 = [BASE1_DIR + i for i in filenames1]
path2 = [BASE2_DIR + i for i in filenames2]


path1 = sorted(path1)
path2 = sorted(path2)

def sisdr(x, s, remove_dc=True):
    """
    Compute SI-SDR
    x: extracted signal
    s: reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm) ** 2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s) ** 2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

if __name__ == '__main__':
    sisdr_i = 0
    pesq_i = 0
    stoi_i = 0

    # sisnr < -20
    a_ = 0
    # -20 <sisnr < -10
    a = 0

    # -10 < sisnr < 0
    b = 0

    # 0 < sisnr < 10
    c = 0

    # 10 < sisnr < 20
    d = 0

    # sisnr >20
    e = 0


    for i in range(2620):

        y1, sr1 = librosa.load(path1[i], sr=8000)
        y2, sr2 = librosa.load(path2[i], sr=8000)
        min_len = min(len(y1), len(y2))
        y1 = y1[0:min_len]
        y2 = y2[0:min_len]
        # y1 = torchaudio.transforms.Tensor(y1)
        # y2 = torchaudio.transforms.Tensor(y2)
        sisdr_k = sisdr(y1, y2)
        pesq_i += pesq(8000, y1, y2, mode='nb')
        stoi_i += stoi(y1, y2, 8000)


        if sisdr_k < -20:
            a_ += 1
        if (sisdr_k >= -20) & (sisdr_k < -10):
            a += 1
        if (sisdr_k >= -10) & (sisdr_k < 0):
            b += 1
        if (sisdr_k >= 0) & (sisdr_k < 10):
            c += 1
        if (sisdr_k >= 10) & (sisdr_k < 20):
            d += 1
        if sisdr_k >= 20:
            e += 1






        print("第" + str(i + 1) + "个: ", sisdr_k, path1[i])
        sisdr_i += sisdr(y1, y2)
    # print('avg of SI-SDR:', sisdr_i / 2620)
    # print(" sisnr < -20的个数为：", a_)
    # print(" -20 <= sisnr < -10的个数为：", a)
    # print(" -10 <= sisnr < 0的个数为：", b)
    # print(" 0 <= sisnr < 10的个数为：", c)
    # print(" 10 <= sisnr < 20的个数为：", d)
    # print(" sisnr >= 20的个数为：", e)
    # print('avg of peaq:', pesq_i / 2620)
    # print('avg of stoi:', stoi_i / 2620)


    print(" sisnr < -20的个数为：", a_)
    print(" -20 <= sisnr < -10的个数为：", a)
    print(" -10 <= sisnr < 0的个数为：", b)
    print(" 0 <= sisnr < 10的个数为：", c)
    print(" 10 <= sisnr < 20的个数为：", d)
    print(" sisnr >= 20的个数为：", e)
    print('SISDR:{:.3f} pesq:{:.3f} stoi:{:.3f}  小于-20:{:.2f}%  大于10:{:.2f}%'.format(sisdr_i / 2620, pesq_i / 2620,
                                                                                     stoi_i / 2620, a_ / 2620 * 100,
                                                                                     (d + e) / 2620 * 100))
