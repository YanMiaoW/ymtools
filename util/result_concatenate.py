import tqdm
import cv2 as cv
from argparse import ArgumentParser
import os
import numpy as np
import glob
from ymlib.common import path_decompose


def parse_args():
    parser = ArgumentParser(description='result concatenate')
    parser.add_argument('-i1', '--one-dir',
                        help='first image  dir', required=True)
    parser.add_argument('-i2','--two-dir',
                        help='second image dir', required=True)
    parser.add_argument('-o', '--output-dir',
                        help='concatenate image save dir', required=True)
    parser.add_argument('--continue-test', action='store_true', help='continue generate')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filepath in tqdm.tqdm(glob.glob(os.path.join(args.image_dir, '*'))):

        _, basename, ext = path_decompose(filepath)
        filename = basename+'.jpg'
        save_path = os.path.join(args.output_dir,filename)
        
        if args.continu and os.path.exists(save_path):
            continue

        if not all([os.path.exists(os.path.join(path, filename)) for path in [args.server_result_dir, args.lollipop_result_dir]]):
            continue

        img_origin = cv.imread(filepath)
        img_server = cv.imread(os.path.join(args.server_result_dir, filename))
        img_lollipop = cv.imread(os.path.join(args.lollipop_result_dir, filename))

        img_origin = cv.resize(img_origin, dsize=(512, 512))
        img_server = cv.resize(img_server, dsize=(512, 512))
        img_lollipop = cv.resize(img_lollipop, dsize=(512, 512))

        img = np.concatenate([img_server, img_lollipop, img_origin], axis=1)
        cv.imencode('.jpg', img)[1].tofile(save_path)
        