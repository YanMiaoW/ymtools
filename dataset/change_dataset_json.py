import glob
import json
import tqdm
from ymlib.common_dataset_api import *
from ymlib.common import get_user_hostname
from ymlib.debug_function import *

if __name__ == '__main__':
    if get_user_hostname() == YANMIAO_MACPRO_NAME:
        dataset_dir = '/Users/yanmiao/yanmiao/data-common/ochuman'
    elif get_user_hostname() == ROOT_201_NAME:
        dataset_dir = '/root/ym/data-common/ochuman'

    for filepath in tqdm.tqdm(glob.glob(os.path.join(dataset_dir, 'data', '*'))):
        with open(filepath, 'r') as f:
            ann = json.load(f)

        objs = ann[key_combine('object', 'sub_list')]

        for obj in objs:
            # print(obj)
            if 'sub_dict::body_keypoints' in obj:
                obj['sub_dict::body_keypoints'] = obj['sub_dict::body_keypoints']
                del obj['sub_dict::body_keypoints']

        ann_json = json.dumps(ann)
        with open(filepath, 'w') as f:
            f.write(ann_json)
