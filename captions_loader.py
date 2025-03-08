from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os

def setup_coco(data_dir, data_type='train2017'):
    """
    Setup COCO dataset
    data_dir: root directory of COCO dataset
    data_type: 'train2017' or 'val2017'
    """
    annotation_file = os.path.join(data_dir, f'captions_{data_type}.json')
    coco = COCO(annotation_file)
    return coco



def test_coco_captions(data_dir, data_type='train2017'):
    """Test COCO caption annotations loading"""
    try:
        # Initialize COCO
        coco = setup_coco(data_dir, data_type)
        # Test annotation loading
        imgIds = coco.getImgIds()
        if len(imgIds) == 0:
            print("Failed: No image IDs found")
            return False

        # Test caption loading
        annIds = coco.getAnnIds(imgIds[0])
        anns = coco.loadAnns(annIds)
        if not anns or 'caption' not in anns[0]:
            print("Failed: No captions found")
            return False
        print("\nSample captions:")
        for img_id in imgIds[:3]:
            annIds = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(annIds)
            print(f"\nImage ID {img_id} captions:")
            for i, ann in enumerate(anns, 1):
                print(f"{i}. {ann['caption']}")

        print(f"Success: Found {len(imgIds)} images")
        print(f"Sample caption: {anns[0]['caption']}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

data_dir = './dataset'
test_coco_captions(data_dir)

