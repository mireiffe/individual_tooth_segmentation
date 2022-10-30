# system libs
from os.path import join

# general libs
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
        
        print(f'image {num_img}, initiating...')

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
        print(f'\t- obtaining pseudo edge-region...')

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
        print(f'\t- refining pre-edge region...')
        img, per0 = self._dt['img'], self._dt['per0']
        
        initC = InitContour(img, per0)
        self._dt.update({
            'phi0': initC.phi0, 'per': initC.per,
            })

        mts.saveFile(self._dt, self.path_dict)
        self.sts.imwrite(initC.per * 255, 'P.png')
        return 0

    def snake(self):
        print(f'\t- contour evolution...')
        img, per, phi0 = self._dt['img'], self._dt['per'], self._dt['phi0']

        SNK = Snake(img, per, phi0)
        phi_res = SNK.snake()
        
        self._dt.update({
            'phi_res': phi_res, 'gadf': SNK.fa, 
            'er': SNK.er,
        })
        mts.saveFile(self._dt, self.path_dict)

        return 0

    def tem(self):
        print(f'\tindentifying regions...')
        img, phi_res = self._dt['img'], self._dt['phi_res']

        IR = TEM(img, phi_res)

        self._dt.update({'lbl_reg': IR.lbl_reg, 'res': IR.res})
        mts.saveFile(self._dt, self.path_dict)

        self._showSaveMax(img, 'result.png', contour=IR.res)

    def _showSaveMax(self, obj, name, face=None, contour=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        resres = []
        ax.imshow(obj)
        if face is not None:
            _res = np.where(face < 0, 0, face)
            plt.imshow(_res, alpha=.4, cmap='rainbow_alpha')
        if contour is not None:
            Rein = Reinitial(dt=.1)
            ReinKapp = ReinitialKapp(iter=5, mu=1)
            clrs = ['lime'] * 100
            for i in range(int(np.max(contour))):
                _reg = np.where(contour == i+1, -1., 1.)
                for _i in range(5):
                    _reg = Rein.getSDF(_reg)
                    _reg = ReinKapp.getSDF(_reg)
                resres.append(_reg)
                plt.contour(_reg, levels=[0], colors=clrs[i], linewidths=2)
            self.sts.savecfg(name)
            plt.close(fig)
        return resres
