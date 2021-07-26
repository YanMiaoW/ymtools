import glob
import json
import tqdm
from ymlib.common_dataset_api import *

if __name__ == '__main__':
    dataset_dir = '/Users/yanmiao/yanmiao/data-common/ochuman'

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
