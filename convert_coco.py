

from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
import shutil

root = "data/Mask_Iron/mask_iron/"
dataroot = ["data/Mask_Iron/mask_iron/coco/", 'data/Mask_Iron/mask_iron/coco1']
splits = ["train", "val"]
for split in splits:
    for dr in dataroot:
        coco = COCO(os.path.join(dr, 'annotations/instance_{}.json'.format(split)))
        img_dir = os.path.join(dr, split)
        len_image_ids = len(os.listdir(img_dir))
        save_path = os.path.join(root, 'convert_annotation/{}'.format(split))
        image_root = os.path.join(dr, split)
        
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "images/"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "labels/"), exist_ok=True)

        for image_id in range(1, len_image_ids+1):
            img = coco.imgs[image_id]
            image_path = os.path.join(image_root, img['file_name'])
            image_name = img['file_name'][:-4] 
            shutil.copyfile(image_path, os.path.join(save_path, "images/" + img['file_name']))
            
            # get annotation
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            coco.showAnns(anns)

            # annotation to mask
            mask = coco.annToMask(anns[0])
            for i in range(1, len(anns)):
                mask = mask + coco.annToMask(anns[i]) # (1080, 1920)
            label = Image.fromarray(mask)
            label.save(os.path.join(save_path, "labels/" + image_name + "_rawlabel.png"))
            label = Image.fromarray(mask*255)
            label.save(os.path.join(save_path, "labels/" + image_name + "_label.png"))
            print('save output image to ', os.path.join(save_path, "labels/" + image_name + "_label.png"))
