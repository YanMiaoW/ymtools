from ymlib.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
from ymlib.dataset_visual import mask2box, draw_box, draw_keypoint, draw_mask, draw_label, index2color
import imgaug as ia
from imgaug import augmenters as iaa
import cv2 as cv
import time
from ymlib.debug_function import *


def show_dataset(dataset_dir):
    aug = iaa.Noop()

    for ann in common_ann_loader(dataset_dir):
        common_choice(ann, key_choices={
                      'image', 'mix', 'segment_mask', 'meta', 'object'})

        def filter(result):
            yield True

        if not common_filter(ann, filter):
            continue

        start = time.time()

        common_transfer(ann)
        common_aug(ann, aug)

        image = ann[key_combine('image', 'image')]
        origin_image_path = ann[key_combine(
            'meta', 'other')]['origin_image_path']

        h, w = image.shape[:2]
        window_name = f'image | mix | mask   height:{h} width:{w} origin:{origin_image_path}  time:{int((time.time() - start)*1000)}'

        # mix = ann[key_combine('mix', 'image')]

        mask = ann[key_combine('segment_mask', 'mask')]

        mix = image.copy()

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

        objs = ann[key_combine('object', 'sub_list')]

        for j0, obj in enumerate(objs):
            common_transfer(obj)
            if key_combine('body_keypoints', 'sub_dict') in obj:
                body_keypoints = obj[key_combine('body_keypoints', 'sub_dict')]
                draw_keypoint(mix, body_keypoints, labeled=True)

            instance_mask = obj[key_combine('instance_mask', 'mask')]

            draw_mask(mix, instance_mask, color=index2color(j0, len(objs)))

        imshow(np.concatenate([image, mix, mask], axis=1), window_name)


if __name__ == '__main__':

    dataset_dir = '/Users/yanmiao/yanmiao/data-common/humanTest'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/hun_sha_di_pian'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/ochuman'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/supervisely'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/coco'

    show_dataset(dataset_dir)
    # test1(dataset_dir)
    # test2()
