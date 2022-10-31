import numpy as np
import matplotlib.pyplot as plt

import src.myTools as mts
import cv2


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

    def calParams(self):
        # self.mu1_i, self.var1_i = distrib(self.img, self.reg1_i)
        # self.mu1_o, self.var1_o = distrib(self.img, self.reg1_o)
        # self.mu2_i, self.var2_i = distrib(self.img, self.reg2_i)
        # self.mu2_o, self.var2_o = distrib(self.img, self.reg2_o)
        delt = 25
        ker = np.ones((delt, delt))
        band = np.where(np.abs(self.phi) < 2, 1., 0.)
        
        eximg = np.expand_dims(self.img, -1)
        img_i = eximg * np.expand_dims(self.reg2_i, 2)
        img_o = eximg * np.expand_dims(self.reg2_o, 2)

        sum_reg_i = cv2.filter2D(self.reg2_i.astype(int), -1, ker)
        sum_reg_o = cv2.filter2D(self.reg2_o.astype(int), -1, ker)
        self.mu3_i = cv2.filter2D(img_i.sum(axis=2), -1, ker) * band / (sum_reg_i*3 + mts.eps)
        self.mu3_o = cv2.filter2D(img_o.sum(axis=2), -1, ker) * band / (sum_reg_o*3 + mts.eps)
        _mu3_i_sq = cv2.filter2D(np.power(img_i, 2).sum(axis=2), -1, ker) * band / (sum_reg_i*3 + mts.eps)
        _mu3_o_sq = cv2.filter2D(np.power(img_o, 2).sum(axis=2), -1, ker) * band / (sum_reg_o*3 + mts.eps)
        self.var3_i = np.abs(_mu3_i_sq - self.mu3_i**2)
        self.var3_o = np.abs(_mu3_o_sq - self.mu3_o**2)

    @staticmethod
    def funPDF(X, mu, sig):
        return np.exp(-(X - mu)**2 / 2 / (sig + mts.eps)**2) / np.sqrt(2 * np.pi) / (sig + mts.eps)

    def force(self):
        _img = self.img.mean(axis=2)[..., None]

        P3_i = self.funPDF(_img, self.mu3_i, self.var3_i**.5)
        P3_o = self.funPDF(_img, self.mu3_o, self.var3_o**.5)
        F3 = np.sign(np.where(self.band, P3_i - P3_o, 0.))

        V2 = (np.where(np.abs(self.mu3_i - self.mu3_o) < .01, 0, F3) * 1/2)

        return V2
