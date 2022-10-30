import numpy as np
import cv2
import skfmm

class Reinitial():
    eps = np.finfo(float).eps

    def __init__(self, dt:float=0.1, width:float=5, tol:float=1E-02, iter:int=None, dim:int=2, debug=False, fmm=False, dim_stack=2) -> np.ndarray:
        '''
        This function make a signed distance function by doing the re-initialization.
        Everything is based on the sussman's paper(1994), but this contains additional 3D implementation.
        https://github.com/mireiffe/reinitialization

        Re-initialization equation: `u_t = sign(u)(1 - |nabla{u}|)`

        ## Inputs
        - dt: scalar, a step size
        - width: scalar, 5 (default)\n
            Numerical error will be computed in region R, R:={x : abs(img(x)) < width}. If None, R = whole domain.
        - tol: scalar, a tolerance
        - iter: scalar,\n
            Number of iterations. if None(default), evolve until touch the tol
        - dim: int,\n
            Dimension of the solution u. If 2 (default), then 3 channels image will be calculated by each channel.

        ## Output
        - phi: numpy array like,\n
            A signed distance function, which means the norm of gradient for the function is 1 (only in the region R).

        ## Referance
        Mark Sussman, Peter Smereka, Stanley Osher
        A Level Set Approach for Computing Solutions to Incompressible Two-Phase Flow
        Journal of Computational Physics, Volume 114, Issue 1, 1994, Pages 146-159.
        https://doi.org/10.1006/jcph.1994.1155
        '''
        self.wid = width
        self.dt = dt
        self.tol = tol
        self.dim = dim
        self.iter = iter
        self.debug = debug
        self.fmm = fmm
        self.dim_stack = dim_stack

    def getSDF(self, img):
        '''
        A main function

        ## Inputs
        - img: numpy array like,\n
            Negative parts are considered as the region of interest.
        '''
        if self.dim_stack == 0:
            img = img.transpose((1, 2, 0))
        if self.fmm:
            if self.dim == 2 and img.ndim == 3:
                phi = np.zeros_like(img)
                for i in range(img.shape[-1]):
                    if len(np.where(img[..., i] < 0)[0]) == 0:
                        phi[..., i] = img[..., i]
                    else:
                        phi[..., i] = skfmm.distance(img[..., i], dx=1)
            else:
                phi = skfmm.distance(img, dx=1)
            if self.dim_stack == 0:
                phi = phi.transpose((2, 0, 1))
            return phi

        self.sign0 = np.sign(img)
        _k = 1
        phi = np.copy(img)
        while True:
            _k += 1
            new_phi = self.update(phi)

            if self.iter != None:
                if _k > self.iter:
                    break
            else:
                if self.wid == None:
                    reg_err = np.ones_like(phi)
                else:
                    reg_err = np.abs(phi) < self.wid
                err = (np.abs(new_phi - phi) * reg_err).sum() / (reg_err.sum() + self.eps)
                if self.debug:
                    print(f"\rk = {_k}, error = {err / self.dt:.6f}", end='')
                if err < self.tol * self.dt:
                    break
            phi = np.copy(new_phi)
        if self.dim_stack == 0:
            new_phi = new_phi.transpose((2, 0, 1))
        return new_phi

    def update(self, phi):
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
        _phi = phi - self.dt * _sign0 * _G
        return _phi

    @staticmethod
    def approx_sign(v:np.ndarray, type=1):
        '''
        ## Inputs
        - v: numpy array like
        - type: 0 or 1 (default),\n
            Type of approximation.\n
            If type == 0:   v / sqrt(v*v + eps*eps) (in the paper),
            elif type == 1: 2/pi * arctan(r * v), r=1000
        '''
        if type == 0:
            _eps = 1
            return v / np.sqrt(v ** 2 + _eps ** 2)
        elif type == 1:
            r = 100
            return 2 / np.pi * np.arctan(r * v)

    @staticmethod
    def grad(v:np.ndarray, dim:int):
        _dx = v[:, 1:, ...] - v[:, :-1, ...]
        _dy = v[1:, :, ...] - v[:-1, :, ...]
        if dim == 2:
            return _dx, _dy
        elif dim == 3:
            _dz = v[:, :, 1:] - v[:, :, :-1]
            return _dx, _dy, _dz
    
    def imgrad(self, v:np.ndarray, dim:int):
        _d = self.grad(v, dim)

        zx = np.zeros_like(v[:, 0:1, ...])
        bx = np.concatenate((zx, _d[0]), axis=1)
        fx = np.concatenate((_d[0], zx), axis=1)

        zy = np.zeros_like(v[0:1, :, ...])
        by = np.concatenate((zy, _d[1]), axis=0)
        fy = np.concatenate((_d[1], zy), axis=0)
        if dim == 2:
            return [bx, by], [fx, fy]
        elif dim == 3:
            zz = np.zeros_like(v[:, :, 0:1])
            bz = np.concatenate((zz, _d[2]), axis=2)
            fz = np.concatenate((_d[2], zz), axis=2)
            return [bx, by, bz], [fx, fy, fz]
        