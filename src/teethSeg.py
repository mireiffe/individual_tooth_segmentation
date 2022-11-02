# system libs
import os
import sys
from os.path import join
import time 

# general libs
import cv2
import numpy as np

# Pytorch
import torch
from torch.utils.data import DataLoader

# morphological imaging libs
from skimage.measure import label
from skimage.morphology import skeletonize

# custom libs
import src.myTools as mts
from .gadf import GADF
from .reinst import ThreeRegions
from .reinitial import Reinitial

from .network import model
from .network.dataset import ErDataset


def measureWidth(per):
    sk_idx = np.where(skeletonize(per) == 1)
    tot_len = len(sk_idx[0])
    np.random.seed(900314)
    sel_idx = np.random.choice(tot_len, tot_len // 10, replace=False)

    wid_er = []
    for si in sel_idx:
        _w = 0
        _x = sk_idx[1][si]
        _y = sk_idx[0][si]
        while True:
            y0 = _y-_w-1 if _y-_w-1 >= 0 else None
            x0 = _x-_w-1 if _x-_w-1 >= 0 else None
            _ptch = per[y0:_y+_w+2, x0:_x+_w+2]
            if _ptch.sum() < _ptch.size:
                wid_er.append(2*_w + 1)
                break
            else:
                _w += 1
    mu = sum(wid_er) / len(sel_idx)
    sig = np.std(wid_er)
    Z_45 = 1.65     # standard normal value for 90 %
    return Z_45 * sig / np.sqrt(tot_len // 10) + mu


class PseudoER():
    def __init__(self, config, num_img):
        self.num_img = num_img
        self.config = config

        self.ROOT = self.config['DEFAULT']['ROOT']
        self.dvc_main = torch.device(config['DEFAULT']['DEVICE'])

    def getEr(self):
        net = self.setModel()
        net.eval()
        with torch.no_grad():
            _img, _er = self.inference(net)
        img = torch.Tensor.cpu(_img).squeeze().permute(1, 2, 0).numpy()
        er = torch.Tensor.cpu(_er).squeeze().numpy()
        return img, er

    def setModel(self):
        kwargs = {
            'in_ch': 3,
            'out_ch': 1,
        }
        net = getattr(model, 'ResNeSt50_TC')(**kwargs).to(device=self.dvc_main)

        if self.dvc_main.type == 'cuda':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.dvc_main.index)
            net = torch.nn.DataParallel(net, device_ids=[self.dvc_main.index])
            print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] + Using main device <{self.dvc_main}>')

            # Load parameters
            file_ld = os.path.join(self.ROOT, self.config['MODEL']['WEIGHTS'])
            checkpoint = torch.load(file_ld, map_location='cpu')
            try:
                net.load_state_dict(checkpoint['net_state_dict'])
            except KeyError:
                net.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            _net = torch.nn.DataParallel(net)
            print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] + Using main device <{self.dvc_main}>')

            # Load parameters
            file_ld = os.path.join(self.ROOT, self.config['MODEL']['WEIGHTS'])
            checkpoint = torch.load(file_ld, map_location='cpu')
            try:
                _net.load_state_dict(checkpoint['net_state_dict'])
            except KeyError:
                _net.load_state_dict(checkpoint['encoder_state_dict'])
            net = _net.module.to(device='cpu')
        net.to(device=self.dvc_main)
        print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] + Model is loaded from {file_ld}')
        return net

    def inference(self, net, dtype=torch.float):
        dir_image = join(self.ROOT, self.config['DATA']['DIR'])
        data_test = ErDataset(dir_image, self.num_img)
        loader_test = DataLoader(
            data_test, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False)

        for k, btchs in enumerate(loader_test):
            imgs = btchs[0].to(device=self.dvc_main, dtype=dtype)
            preds = net(imgs)

            om, on = data_test.m, data_test.n
            m, n = imgs.shape[2:4]
            mi = (m - om) // 2
            ni = (n - on) // 2
        return imgs[..., mi:mi + om, ni:ni + on], preds[..., mi:mi + om, ni:ni + on]


class InitContour():
    def __init__(self, img:np.ndarray, per0:np.ndarray) -> np.ndarray:
        self.img = img
        self.per0 = per0
        self.m, self.n = self.per0.shape
        self.preset()
        self.per = mts.imDilErod(
            self.per, rad=max(round(self.wid_er / 2), 1),
            kernel_type='circular')

        rein_all = Reinitial(width=None, fmm=True)
        # Setting fmm to True can speed up the evolution process
        self.rein_w5 = Reinitial(width=5, dim_stack=0, fmm=False)
        self.rein_w5_fmm = Reinitial(width=5, dim_stack=0, fmm=True)

        # get landmarks
        self.phi_per = rein_all.getSDF(self.per - .5)
        lmk = self.getLandMarks(self.phi_per)
        self.phi_lmk = self.rein_w5_fmm.getSDF(.5 - (lmk[np.newaxis, ...] < 0))

        # bring them back
        self.phi_back = self.bringBack(self.phi_lmk, self.per, gap=8, dt=.3, mu=2, nu=1, reinterm=3, tol=2, max_iter=500)

        # separate level sets 
        reg_sep = self.sepRegions(self.phi_back)
        phi_sep = self.rein_w5.getSDF(.5 - np.array(reg_sep))

        # initials
        phi_init = self.evolve(phi_sep, self.per, dt=.3, mu=2, nu=1, reinterm=3, tol=2, max_iter=100)
        self.phi0 = phi_init
        return

    def preset(self):
        self.wid_er = measureWidth(self.per0)
        self.per = np.copy(self.per0)

    def getLandMarks(self, phi0):
        m_phi = np.where(mts.local_minima(phi0), phi0, 0)
        phi = np.copy(phi0)
        while True:
            _lbl = label(phi < 0)
            _reg = np.zeros_like(_lbl)
            for l in np.unique(_lbl)[1:]:
                _r = np.where(_lbl == l, True, False)
                if np.min(phi * _r) <= np.min(m_phi*_r) / 2:
                    _reg += _r
            if _reg.sum() == 0:
                break
            phi += _reg
        return phi

    def sepRegions(self, phi_back):
        lbl_per = label(self.per, background=1, connectivity=1)
        lbl_back = label(phi_back[0] < 0, background=0, connectivity=1)
        reg_sep = [np.zeros_like(phi_back[0]), ]
        for l in np.unique(lbl_per)[1:]:
            _backs = np.setdiff1d(np.unique(lbl_back * (lbl_per == l)), [0])
            if len(reg_sep) < len(_backs):
                for _ in range(len(_backs) - len(reg_sep)):
                    reg_sep.append(np.zeros_like(phi_back[0]))
            for _i, _l in enumerate(_backs):
                reg_sep[_i] += np.where(lbl_back == _l, 1., 0.)
        return reg_sep

    def evolve(self, phi, wall, dt, mu, nu, reinterm, tol, max_iter):
        phi0 = np.copy(phi)
        k = 0
        dist = 1

        err = []
        while True:
            regs = np.where(phi < dist, phi - dist, 0)
            all_regs = regs.sum(axis=0)
            Fc = (- (all_regs - regs) - 1)

            kapp = mts.kappa(phi0, stackdim=0)[0]
            phi = phi0 + dt * ( (Fc + mu * kapp) * (1 - wall) + (nu * wall) )
            if k % 10 == 0: print('.', end=''); sys.stdout.flush()

            reg0 = np.where(phi0 < 0, 1, 0)
            reg = np.where(phi < 0, 1, 0)
            err.append((reg0 + reg - 2 * reg0 * reg).sum()) 
            if len(err) > 4:
                if (sum(err)  < tol) or (k > max_iter):
                    break
                err.pop(0)
            if k % reinterm == 0:
                phi = self.rein_w5_fmm.getSDF(np.where(phi < 0, -1., 1.))

            k += 1
            phi = mts.remove_pos_lvset(phi)[0]
            phi0 = phi
        print('')
        phi = self.rein_w5.getSDF(np.where(phi < 0, -1., 1.))
        return phi

    def bringBack(self, phi, per, gap, dt, mu, nu, reinterm, tol, max_iter):
        wall = mts.gaussfilt(cv2.dilate(per, np.ones((2*gap + 1, 2*gap + 1))), sig=1)
        lbl_per = label(per, background=1, connectivity=1)
        for l in np.unique(lbl_per)[1:]:
            _r = np.where(lbl_per == l)
            if (wall > 0.01)[_r].sum() == len(_r[0]):
                wall[_r] = 1/(2-nu)

        wall = np.where(phi < .1, 0, wall)
        phi0 = np.copy(phi)
        k = 0

        err = [999]
        print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] + Expanding contours to boundary of the pseudo edge-region', end='')
        while True:
            kapp = mts.kappa(phi0, stackdim=0)[0]
            phi = phi0 + dt * ( (-1 + mu * kapp) * (1 - wall)) + wall
            if k % 10 == 0: print('.', end=''); sys.stdout.flush()

            # error estimation
            reg0 = np.where(phi0 < 0, 1, 0)
            reg = np.where(phi < 0, 1, 0)
            err.append((reg0 + reg - 2 * reg0 * reg).sum()) 
            if len(err) > 4:
                if (sum(err)  < tol) or (k > max_iter):
                    break
                err.pop(0)

            if k % reinterm == 0:
                # phi = self.rein_w5_fmm.getSDF(np.where(phi < 0.2, -1., 1.))
                phi = self.rein_w5_fmm.getSDF(np.where(phi < 0.2, -1., 1.))
            k += 1
            phi0 = phi
        phi = self.rein_w5.getSDF(np.where(phi < 0.2, -1., 1.))
        return phi


class Snake():
    def __init__(self, img:np.ndarray, per:np.ndarray, phi0) -> None:
        self.img = img
        self.per = per
        self.m, self.n = self.per.shape

        # find GADF for gray img
        GF = GADF(self.img.mean(axis=2))
        self.fa = GF.Fa
        self.er = GF.Er

        self.wid_er = measureWidth(self.per)
        self.rein = Reinitial(dt=.2, width=3, tol=0.05, dim_stack=0)

        self.phi0 = self.sepRegions(phi0)

    def sepRegions(self, phi):
        _rein = Reinitial(dt=.2, width=self.wid_er * 3, tol=0.01, fmm=True)
        lbl_phi = np.zeros_like(phi[0])
        for ph in phi:
            _lbl = label(np.where(ph < 0, 1., 0.), background=0, connectivity=1)
            lbl_phi = np.where(_lbl > .5, _lbl + lbl_phi.max(), lbl_phi)
        phis = []
        for l in np.unique(lbl_phi)[1:]:
            _reg = np.where(lbl_phi == l, 1., 0.)
            phis.append(_rein.getSDF(.5 - _reg))

        return np.array(phis)

    def snake(self, dt=0.2, mu=2, tol=2, dist=1, max_iter=30, reinterm=5):
        n_phis = len(self.phi0)
        tega = ThreeRegions(self.img)
        phis = np.copy(self.phi0)

        stop_reg = np.ones_like(self.per)
        stop_reg[2:-2, 2:-2] = 0
        
        self.use_er = self.er * ((phis > -1).sum(axis=0) == n_phis)
        oma = self.use_er
        omc = (1 - oma) * (1 - stop_reg)
        oms = (1 - oma) * (1 - stop_reg)

        k = 0
        err = [999]
        while True:
            k += 1
            if k % reinterm == 0:
                phis = self.rein.getSDF(np.where(phis < 0, -1., 1.))

            regs = np.where(phis < dist, phis - dist, 0)
            all_regs = regs.sum(axis=0)
            Fc = (- (all_regs - regs) - 1)

            gx, gy = mts.imgrad(phis.transpose((1, 2, 0)))
            Fa = - (gx.transpose((2, 0, 1)) * self.fa[..., 1] + gy.transpose((2, 0, 1)) * self.fa[..., 0])
            tega.setting(phis.transpose(1, 2, 0))
            _Fs = -tega.force().transpose(2, 0, 1)
            Fs = mts.gaussfilt(_Fs, sig=1, stackdim=0)

            kap = mts.kappa(phis, stackdim=0)[0]
            F = Fa*oma + Fs*oms + Fc*omc + mu*kap
            new_phis = phis + dt * F
            if k % 10 == 0: print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] + iter: {k}')

            reg0 = np.where(phis < 0, 1, 0)
            reg = np.where(new_phis < 0, 1, 0)
            err.append((reg0 + reg - 2 * reg0 * reg).sum()) 
            if len(err) > 4:
                if (sum(err)  < tol) or (k > max_iter):
                    break
                err.pop(0)
            new_phis = mts.remove_pos_lvset(new_phis)[0]
            n_phis = len(new_phis)
            phis = new_phis
        phis = self.rein.getSDF(np.where(phis < 0, -1., 1.))
        return new_phis


class TEM():
    def __init__(self, img:np.ndarray, phi_res) -> None:
        self.img = img
        self.phi_res = phi_res

        self.m, self.n = self.img.shape[:2]
        self.lbl_reg = self.setReg(self.phi_res)
        self.res = self.regClass()

    def setReg(self, phis):
        res = -np.ones((self.m, self.n))
        for i, phi in enumerate(phis):
            if len(np.where(phi < 0)[0]) > self.m * self.n / 500:
                res = np.where(phi < 0, i, res)
        return res

    def regClass(self):
        lbl_inreg = self.removeBG(self.lbl_reg)
        lbl_thres = self.thresEnhanced(self.img, lbl_inreg)
        lbl_sd = self.removeSide(self.img, lbl_thres)
        return lbl_sd

    def thresEnhanced(self, img, lbl, mu=0.3):
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        en_img = np.where(R < 1E-01, 0, G / (R + mts.eps)) - mu*B

        dist_reg = en_img >= 1E-03

        thres = en_img > np.mean(en_img, where=dist_reg) + 0.0 * np.std(en_img, where=dist_reg)

        res = np.zeros_like(lbl)
        for l in np.unique(lbl[1:]):
            _r = (lbl == l)
            if (_r * thres).sum() > .75 * _r.sum():
                res = np.where(_r, l, res)

        return res

    @staticmethod
    def removeBG(lbl):
        tnb = np.unique(lbl[[0, -1], :])
        lnr = np.unique(lbl[:, [0, -1]])

        res = np.copy(lbl)
        for l in np.unique(np.union1d(tnb, lnr)):
            res = np.where(lbl == l, -1., res)
        return res

    @staticmethod
    def removeSide(img, lbl):
        idx = np.where(lbl > 0)
        _r = np.argmax(idx[1])
        _l = np.argmin(idx[1])

        R = lbl[idx[0][_r], idx[1][_r]]
        L = lbl[idx[0][_l], idx[1][_l]]

        res = np.copy(lbl)
        mu = np.mean(img)
        sig = np.sqrt(np.var(img))
        for i in [R, L]:
            _reg = (lbl == i)
            _mu = np.mean(img.transpose((2, 0, 1)), where=_reg)
            if _mu < mu - 1 * sig:
                res = np.where(_reg, -1, res)
        return res


class ReinitialKapp(Reinitial):
    def __init__(self, dt:float=0.1, width:float=5, tol:float=1E-02, iter:int=None, dim:int=2, debug=False, fmm=False, mu=0.1) -> np.ndarray:
        super().__init__(dt=dt, width=width, tol=tol, iter=iter, dim=dim, debug=debug, fmm=fmm)
        self.mu = mu

    def update(self, phi):
        m, n = phi.shape[:2]

        bd, fd = self.imgrad(phi, self.dim)

        # abs(a) and a+ in the paper
        bxa, bxp = np.abs(bd[0]), np.maximum(bd[0], 0)
        # abs(b) and b+ in the paper
        fxa, fxp = np.abs(fd[0]), np.maximum(fd[0], 0)
        # abs(c) and c+ in the paper
        bya, byp = np.abs(bd[1]), np.maximum(bd[1], 0)
        # abs(d) and d+ in the paper
        fya, fyp = np.abs(fd[1]), np.maximum(fd[1], 0)
        if self.dim == 3:
            bza, bzp = np.abs(bd[2]), np.maximum(bd[2], 0)
            fza, fzp = np.abs(fd[2]), np.maximum(fd[2], 0)

        b_sgn, f_sgn = (self.sign0 - 1) / 2, (self.sign0 + 1) / 2

        Gx = np.maximum((bxa * b_sgn + bxp) ** 2, (-fxa * f_sgn + fxp) ** 2)
        Gy = np.maximum((bya * b_sgn + byp) ** 2, (-fya * f_sgn + fyp) ** 2)
        if self.dim == 2:
            _G = np.sqrt(Gx + Gy) - 1
        elif self.dim == 3:
            Gz = np.maximum((bza * b_sgn + bzp) ** 2, (-fza * f_sgn + fzp) ** 2)
            _G = np.sqrt(Gx + Gy + Gz) - 1
        
        # for numerical stabilities, sign should be updated
        _sign0 = self.approx_sign(phi)
        kapp = mts.gaussfilt(mts.kappa(phi)[0], sig=np.ceil(m*n/300000))
        _phi = phi - self.dt * (_sign0 * _G - self.mu * kapp)
        return _phi
