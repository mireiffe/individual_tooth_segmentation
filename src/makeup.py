# system libs
from os.path import join
import time 

# general libs
import cv2
import numpy as np
import matplotlib.pyplot as plt

# custom libs
import src.myTools as mts
from .reinitial import Reinitial
from .teethSeg import PseudoER, InitContour, Snake, TEM, ReinitialKapp

# global variables
jet_alpha = mts.colorMapAlpha(plt)
brg_alpha = mts.colorMapAlpha(plt, cmap='brg')


class TeethSeg():
    def __init__(self, dir_result, num_img, sts:mts.SaveTools, config) -> None:
        self.num_img = num_img
        self.sts = sts
        self.config = config
        self.maxlen = 500
        
        print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] Image {num_img}, initiating...')

        # set an empty dictionary
        self.path_dict = join(dir_result, f'{num_img:06d}.pth')
        try:
            self._dt:dict = mts.loadFile(self.path_dict)
        except FileNotFoundError:
            self._dt = {}

    def pseudoER(self):
        '''
        Inference of a pseudo edge-region using a deep neural net
        '''
        print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] Step1: Obtaining pseudo edge-region...')

        PER = PseudoER(self.config, self.num_img)
        inpt, oupt = PER.getEr()             # network inference
        per0 = np.where(oupt > .5, 1., 0.)     # pseudo edge-region

        self._dt.update({
            'img': inpt, 'output': oupt, 'per0': per0
        })
        mts.saveFile(self._dt, self.path_dict)

        # saving images
        self.sts.imwrite(inpt * 255, 'input.png')
        self.sts.imwrite(oupt * 255, 'output.png')

        return 0

    def initContour(self):
        print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] Step2: Refining pre-edge region...')
        img, per0 = self._dt['img'], self._dt['per0']
        
        if self.config['RESIZE']:
            m, n = img.shape[:2]
            if n > self.maxlen:
                img = self._resize(img)
                per0 = self._resize(per0)
        initC = InitContour(img, per0)
        if self.config['RESIZE']:
            if n > self.maxlen:
                initC.per = self._resize(initC.per, restore=(m, n))
                initC.phi0 = self._resize(initC.phi0.transpose(1, 2, 0), restore=(m, n)).transpose(2, 0, 1)
        self._dt.update({
            'phi0': initC.phi0, 'per': initC.per,
            })

        mts.saveFile(self._dt, self.path_dict)
        self.sts.imwrite(initC.per * 255, 'P.png')
        return 0

    def snake(self):
        print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] Step3: Contour evolution...')
        img, per, phi0 = self._dt['img'], self._dt['per'], self._dt['phi0']

        if self.config['RESIZE']:
            m, n = img.shape[:2]
            if n > self.maxlen:
                img = self._resize(img)
                per = self._resize(per)
                phi0 = self._resize(phi0.transpose(1, 2, 0), rein=True).transpose(2, 0, 1)
        SNK = Snake(img, per, phi0)
        phi_res = SNK.snake()
        if self.config['RESIZE']:
            if n > self.maxlen:
                phi_res = self._resize(phi_res.transpose(1, 2, 0), restore=(m, n)).transpose(2, 0, 1)
                SNK.fa = self._resize(SNK.fa, restore=(m, n))
                SNK.er = self._resize(SNK.er.astype(float), restore=(m, n))
        self._dt.update({
            'phi_res': phi_res, 'gadf': SNK.fa, 
            'er': SNK.er,
        })
        mts.saveFile(self._dt, self.path_dict)

        return 0

    def tem(self):
        print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] Step4: Indentifying regions...')
        img, phi_res = self._dt['img'], self._dt['phi_res']
        IR = TEM(img, phi_res)

        final = self._showSaveMax(img, 'result', contour=IR.res)
        self._dt.update({'lbl_reg': IR.lbl_reg, 'res': IR.res, 'final': final})
        mts.saveFile(self._dt, self.path_dict)

    def _showSaveMax(self, obj, name, face=None, contour=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        res = []
        ax.imshow(obj)
        if face is not None:
            _res = np.where(face < 0, 0, face)
            plt.imshow(_res, alpha=.7, cmap=mts.rainbow_alpha)
        if contour is not None:
            Rein = Reinitial(dt=.2, width=2, tol=0.05)
            ReinKapp = ReinitialKapp(iter=5, mu=2, tol=0.0001)
            clrs = ['lime'] * 100
            for i in range(int(np.max(contour))):
                _reg = np.where(contour == i+1, -1., 1.)
                for _i in range(5):
                    _reg = Rein.getSDF(_reg)
                    _reg = ReinKapp.getSDF(_reg)
                res.append(_reg)
                plt.contour(_reg, levels=[0], colors=clrs[i], linewidths=1.5)
        self.sts.savecfg(name + '.png')
        plt.close(fig)
        return res

    def _resize(self, x, rein=False, restore=False):
        if restore:
            res = cv2.resize(x, (restore[1], restore[0]))
        else:
            m, n = x.shape[:2]
            if n > m:
                res = cv2.resize(x, (self.maxlen, round(m/n*self.maxlen)))
            else:
                res = cv2.resize(x, (round(n/m*self.maxlen), self.maxlen))
        if rein:
            _rein = Reinitial(width=3, dim_stack=2)
            res = _rein.getSDF(res)
        return res