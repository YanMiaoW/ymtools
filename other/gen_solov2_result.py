from posixpath import basename
import numpy as np
import requests
import json
import os
import cv2 as cv
from argparse import ArgumentParser
import tqdm


def parse_args():
    parser = ArgumentParser(description='solov2 test image dir')
    parser.add_argument('-i', '--test-image-dir',
                        help='image test dir', required=True)
    parser.add_argument('-o', '--output-dir',
                        help='image save dir', required=True)
    parser.add_argument('-t', '--type', help='img type choice from (mask, overlay)',default='overlay')
    parser.add_argument('--continu', action='store_true', help='continue generate')
    args = parser.parse_args() 
    return args


def path_decompose(path):
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    ext = os.path.splitext(path)[-1]
    basename = os.path.splitext(basename)[0]
    return dirname, basename, ext


def request_mask(img_path, save_path, out_type):
    img = cv.imread(img_path)
    if img is None:
        return
    img = cv.resize(img, (512, 512))
    _, _, ext = path_decompose(img_path)
    if ext not in ['.jpg', '.png','.jpgerr']:
        print('find ' + img_path + ' not jpg or png or else')
        return
    _, img_encode = cv.imencode('.JPEG' if (ext == '.jpg' or ext == '.jpgerr') else '.PNG', img)

    url = "http://47.97.38.28:8088/handle"
    files = {'file': img_encode}
    try:
        ret = requests.post(url, files=files, stream=True)
    except:
        return 
    if ret.status_code == 200:
        # with open(save_path, 'w+') as f:
        #     f.write(ret.text)
        byte_encode = ret.raw.read()
        img_encode = np.asarray(bytearray(byte_encode), dtype="uint8")
        mask_img = cv.imdecode(img_encode, cv.IMREAD_COLOR)
        
        if out_type == 'mask':
            cv.imencode('.png', mask_img)[1].tofile(save_path)
        elif out_type =='overlay':
            # res_img = np.bitwise_and(img,  mask_img)
            # mask_img_2 = np.bitwise_and(img, 255 - mask_img) //4 + (255 - mask_img)*[0,0,1] // 4 *3
            # out_img = res_img + mask_img_2
            
            res = mask_img[:,:] != [255, 255, 255]
            res = res.max(2)
            
            img[res] = mask_img[res] // 2 + img[res] // 2
            out_img = img
            cv.imencode('.jpg', out_img)[1].tofile(save_path)
            
        # print('save file ' + filename)
    else:
        print('faial, response code is ' + str(ret.status_code))
        return


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in tqdm.tqdm(os.listdir(args.test_image_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpgerr"):

            filepath = os.path.join(args.test_image_dir, filename)
            
            _, basename, ext = path_decompose(filepath)
            
            save_path = os.path.join(args.output_dir, basename+'.jpg')
            
            if args.continu and os.path.exists(save_path):
                continue

            request_mask(filepath, save_path, args.type)
