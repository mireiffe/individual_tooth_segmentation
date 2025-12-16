import os
from os.path import join

import cv2
import pickle
import numpy as np

import matplotlib.pyplot as plt
from numpy.lib.function_base import iterable
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

eps = np.finfo(float).eps

def saveFile(dict:dict, path:str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(dict, f)
    return 0

def loadFile(path:str) -> None:
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def makeDir(path) -> None:
    try:
        os.mkdir(path)
        print(f"Created a directory {path}")
    except OSError:
        pass

def imwrite(img, path):
    img = np.clip(img, a_min=0, a_max=1)
    plt.imsave(path, img)

def grad(v:np.ndarray, dim:int, method:str):
    '''
    Output
    ---
    d = dy, dx, dz
    '''
    if method in {'forward', 'backward'}:
        h = 1
    elif method in {'central'}:
        h = 2
    dx = v[:, h:, ...] - v[:, :-h, ...]
    dy = v[h:, :, ...] - v[:-h, :, ...]
    if dim == 2:
        return dy / h, dx / h
    elif dim == 3:
        dz = v[:, :, h:] - v[:, :, :-h]
        return dy / h, dx / h, dz / h

def imgrad(v:np.ndarray, dim:int=2, order=1, method='central'):
    '''
    Output
    ---
    d = dy, dx, dz if order==1,
        dyy, dxx, dxy if order==2.
    '''
    if order == 1:
        _d = grad(v, dim, method)
        d = []
        for i in range(dim):
            sz = list(v.shape)
            sz[i] = 1
            _z = np.zeros(sz)
            if method in {'forward'}:
                d.append(np.concatenate((_d[i], _z), axis=i))
            elif method in {'backward'}:
                d.append(np.concatenate((_z, _d[i]), axis=i))
            elif method in {'central'}:
                d.append(np.concatenate((_z, _d[i], _z), axis=i))
        return d
    elif order == 2:
        _v = np.pad(v, ((1, 1), (1, 1), (0, 0)), mode='symmetric')
        dxx = _v[1:-1, 2:, ...] + _v[1:-1, :-2, ...] - 2 * _v[1:-1, 1:-1, ...]
        dyy = _v[2:, 1:-1, ...] + _v[:-2, 1:-1, ...] - 2 * _v[1:-1, 1:-1, ...]
        dxy = _v[2:, 2:, ...] + _v[:-2, :-2, ...] - _v[2:, :-2, ...] - _v[:-2, 2:, ...]
        return dyy, dxx, dxy

def gaussfilt(img:np.ndarray, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT, stackdim=2):
    if stackdim == 0: img = img.transpose((1, 2, 0))
    if ksz is None:
        ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
    res = cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)
    if stackdim == 0: res = res.transpose((2, 0, 1))
    return res

def kappa(phis, mode=0, stackdim=2):
    if stackdim == 0: phis = phis.transpose((1, 2, 0))
    y, x = imgrad(phis, dim=2, order=1, method='central')
    if mode == 0:
        ng = np.sqrt(x**2 + y**2 + eps)
        nx, ny = x / ng, y / ng
        _, xx = imgrad(nx)
        yy, _ = imgrad(ny)
        res = xx + yy
    elif mode == 1:
        yy, xx, xy = imgrad(phis, dim=2, order=2, method='central')
        den = xx * y * y - 2 * x * y * xy + yy * x * x
        num = np.power(x ** 2 + y ** 2, 1.5)
        ng = np.sqrt(x**2 + y**2 + eps)        # just for output
        res = den / (num + eps)
    if stackdim == 0: 
        res = res.transpose((2, 0, 1))
        x = x.transpose((2, 0, 1))
        y = y.transpose((2, 0, 1))
        ng = ng.transpose((2, 0, 1))
    return res, x, y, ng

def cker(rad):
    rad = np.maximum(round(rad), 1)
    Y, X = np.indices([2 * rad + 1, 2 * rad + 1])
    cen_pat = rad
    return np.where((X - cen_pat)**2 + (Y - cen_pat)**2 <= rad**2, 1., 0.)

def imDilErod(img, rad, kernel_type='circular'):
    if kernel_type == 'circular':
        ker = cker(rad)
    elif kernel_type == 'rectangular':
        ker = np.ones((2*rad + 1, 2*rad + 1))

    res_dil = np.where(cv2.filter2D(img, -1, kernel=ker) > 1E-03, 1., 0.)
    res = np.where(1 - cv2.filter2D(1 - res_dil, -1, kernel=ker) < 1E-03, 0., 1.)

    return res

def remove_pos_lvset(phi: np.ndarray, *cormvs):
    lst_rmv = []
    for iph, ph in enumerate(phi):
        if (ph < 0).sum() < 3:
            lst_rmv.append(iph)
    res = [np.delete(phi, lst_rmv, axis=0)]
    for cr in cormvs:
        res.append(np.delete(cr, lst_rmv, axis=0))
    return res

def local_minima(arr):
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    background = (arr==0)
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    return detected_minima

def sortEig(X):
    d, q = np.linalg.eig(X)
    sd = np.argsort(d)[::-1]
    return d[sd], np.take_along_axis(q, np.stack((sd, sd)), axis=1)

class SaveTools():
    def __init__(self, dir_save) -> None:
        self.dir_save = dir_save

    def saveFile(self, dict:dict, name) -> None:
        with open(os.path.join(self.dir_save, name), 'wb') as f:
            pickle.dump(dict, f)
        return 0

    def imwrite(self, img, name_save):
        if img.ndim == 3:
            img = img[..., ::-1]
        cv2.imwrite(join(self.dir_save, name_save), img)

    def imshow(self, img, name_save, cmap=None):
        fig = plt.figure()
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.savefig(join(self.dir_save, name_save), dpi=256, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def imshows(self, imgs: iterable, name_save: str, cmaps: iterable, alphas: iterable):
        fig = plt.figure()
        for im, cm, alph in zip(*[imgs, cmaps, alphas]):
            plt.imshow(im, cmap=cm, alpha=alph)
        plt.axis('off')
        plt.savefig(join(self.dir_save, name_save), dpi=256, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def imcontour(self, img, contours: iterable, name_save: str, cmap=None, colors='lime'):
        fig = plt.figure()
        plt.imshow(img, cmap=cmap)
        for ctr in contours:
            plt.contour(ctr, levels=[0], colors=colors, linewidths=1.3)
        plt.axis('off')
        plt.savefig(join(self.dir_save, name_save), dpi=256, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def savecfg(self, name_save):
        plt.axis('off')
        plt.savefig(join(self.dir_save, name_save), dpi=256, bbox_inches='tight', pad_inches=0)


def colorMapAlpha(_plt, cmap='jet', _name='alpha') -> None:
    name = cmap + _name
    # get colormap
    ncolors = 256
    color_array = _plt.get_cmap(cmap)(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
    # create a colormap object
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib
    map_object = LinearSegmentedColormap.from_list(name=name, colors=color_array)
    # register this new colormap with matplotlib (only if not already registered)
    if name not in matplotlib.colormaps:
        matplotlib.colormaps.register(cmap=map_object)
    return name

jet_alpha = colorMapAlpha(plt, cmap='jet')
rainbow_alpha = colorMapAlpha(plt, cmap='rainbow')
grainbow_alpha = colorMapAlpha(plt, cmap='gist_rainbow')


if __name__ == '__main__':
    pass