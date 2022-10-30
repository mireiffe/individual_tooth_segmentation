import numpy as np
import matplotlib.pyplot as plt

import src.myTools as mts
import time


def distrib(img:np.ndarray, reg):
    if img.ndim == 3:
        img = img.transpose((2, 0, 1))
    mu = img.mean(where=reg)
    var = img.var(where=reg)
    return mu, var

class ThreeRegions():
    def __init__(self, img) -> None:
        self.img = img
        self.m, self.n = self.img.shape[:2]
        self.n_ch = self.img.shape[-1]

    def setting(self, phi):
        self.phi = phi

        self.globalReg()
        self.bandReg()
        self.calParams()

    def globalReg(self):
        self.reg1_i = self.phi < 0
        self.reg1_o = self.phi > 0

    def bandReg(self, gamma=7):
        self.band = np.abs(self.phi) < gamma
        self.reg2_i = self.reg1_i * self.band
        self.reg2_o = self.reg1_o * self.band

    def localReg(self, x, y, delta=25):
        _d = delta // 2
        y1 = np.clip(y - _d, 0, self.m)
        x1 = np.clip(x - _d, 0, self.n)

        y2 = np.clip(y + _d + 1, 0, self.m)
        x2 = np.clip(x + _d + 1, 0, self.n)

        return x1, x2, y1, y2

    def calParams(self):
        # self.mu1_i, self.var1_i = distrib(self.img, self.reg1_i)
        # self.mu1_o, self.var1_o = distrib(self.img, self.reg1_o)
        # self.mu2_i, self.var2_i = distrib(self.img, self.reg2_i)
        # self.mu2_o, self.var2_o = distrib(self.img, self.reg2_o)

        self.mu3_i, self.var3_i = np.zeros_like(self.phi), np.zeros_like(self.phi)
        self.mu3_o, self.var3_o = np.zeros_like(self.phi), np.zeros_like(self.phi)
        
        idx_band = np.where(np.abs(self.phi) < 2)
        img_i = self.img.transpose((2, 0, 1)) * self.reg2_i
        img_o = self.img.transpose((2, 0, 1)) * self.reg2_o
        for y, x in zip(*idx_band):
            x1, x2, y1, y2 = self.localReg(x, y)
            # _img = self.img[y1:y2, x1:x2, ...].transpose((2, 0, 1))
            # _reg_i = self.reg2_i[y1:y2, x1:x2]
            # _reg_o = self.reg2_o[y1:y2, x1:x2]
            _img_i = img_i[..., y1:y2, x1:x2]
            _img_o = img_o[..., y1:y2, x1:x2]

            if _img_i.sum() == 0:
                self.mu3_i[y, x] = 0
                self.var3_i[y, x] = 0
            else:
                # self.mu3_i[y, x] = _img_i.mean(where=_reg_i)
                # self.var3_i[y, x] = _img_i.var(where=_reg_i)
                self.mu3_i[y, x] = _img_i.mean()
                self.var3_i[y, x] = _img_i.var()
            if _img_o.sum() == 0:
                self.mu3_o[y, x] = 0
                self.var3_o[y, x] = 0
            else:
                self.mu3_o[y, x] = _img_o.mean()
                self.var3_o[y, x] = _img_o.var()

    def force(self):
        def funPDF(X, mu, sig):
            return np.exp(-(X - mu)**2 / 2 / (sig + mts.eps)**2) / np.sqrt(2 * np.pi) / (sig + mts.eps)

        _img = self.img.mean(axis=2)

        P3_i = funPDF(_img, self.mu3_i, self.var3_i**.5)
        P3_o = funPDF(_img, self.mu3_o, self.var3_o**.5)
        F3 = np.sign(np.where(self.band, P3_i - P3_o, 0.))

        V2 = (np.where(np.abs(self.mu3_i - self.mu3_o) < .01, 0, F3) * 1/2)

        return V2
