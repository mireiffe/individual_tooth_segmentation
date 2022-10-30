# system libs
import os
import time
from os.path import join, splitext

# libs
import argparse
import yaml

# custom libs
import src.myTools as mts
from src.makeup import TeethSeg

# global variables
today = time.strftime("%y-%m-%d", time.localtime(time.time()))

def get_args():
    parser = argparse.ArgumentParser(description='Individual tooth segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--pseudo_er", dest="pseudo_er",
                             required=False, action='store_true',
                             help="Network inference making pseudo edge region")
    parser.add_argument("-c", "--init_contours", dest="inits",
                             required=False, action='store_true',
                             help="Obtain initial contours")
    parser.add_argument("-s", "--snake", dest="snake",
                             required=False, action='store_true',
                             help="Snake; active contour evolution")
    parser.add_argument("-i", "--id_region", dest="id_region",
                             required=False, action='store_true',
                             help="Identification of regions")
    parser.add_argument("-A", "--ALL", dest="ALL",
                             required=False, action='store_true',
                             help="Do every process in a row")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default='config/default.yaml',
                             required=False, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.ALL:
        args.pseudo_er = True
        args.inits = True
        args.snake = True
        args.id_region = True
    with open(args.path_cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ROOT = config['DEFAULT']['ROOT']
    dir_image = join(ROOT, config['DATA']['DIR'])
    dir_output = join(ROOT, *[config['EVAL']['DIR'], f'{today}/'])
    mts.makeDir(join(ROOT, config['EVAL']['DIR']))
    mts.makeDir(dir_output)

    imgs = [int(splitext(file)[0]) for file in os.listdir(dir_image) if splitext(file)[-1][1:] in config['DATA']['EXT']]

    for ni in imgs:
        dir_img = join(dir_output, f'{ni:05d}/')
        mts.makeDir(dir_img)
        sts = mts.SaveTools(dir_img)
        
        # Inference pseudo edge-regions with a deep neural network
        ts = TeethSeg(dir_img, ni, sts, config)
        if args.pseudo_er: ts.pseudoER()
        if args.inits: ts.initContour()
        if args.snake: ts.snake()
        if args.id_region: ts.tem()
    