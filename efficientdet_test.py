# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import yaml
import argparse
import glob
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.autonotebook import tqdm

from evaluator import *
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess


def display(preds, imgs, imshow=True, imwrite=False, obj_list=[], compound_coef=0):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('-m', '--model_path', type=str, help='model path')
    parser.add_argument('--img_dir', type=str, help='img dir')
    parser.add_argument('--ann_file', type=str, help='ground truth file')

    args = parser.parse_args()
    return args


def test(opt):
    params = Params(f'projects/{opt.project}.yml')
    project_name = params.project_name
    obj_list = params.obj_list
    compound_coef = opt.compound_coef
    force_input_size = None  # set None to use default size
    img_dir = opt.img_dir
    model_path = opt.model_path

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    gt = COCO(opt.ann_file)
    gt_lst = load_coco_bboxes(gt, is_gt=True)

    imgs = glob.glob(os.path.join(img_dir, '*.jpg'))
    det_lst = []
    progressbar = tqdm(imgs)
    for i, img in enumerate(progressbar):
        det = single_img_test(img, input_size, model, use_cuda, use_float16)
        det_lst.extend(det)
        progressbar.update()
        progressbar.set_description('Step: {}/{}'.format(i, len(imgs)))


    evaluator = Evaluator()
    ret, mAP = evaluator.GetMAPbyClass(
        gt_lst,
        det_lst,
        method='EveryPointInterpolation'
    )
    # Get metric values per each class
    for metricsPerClass in ret:
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        ap_str = '{0:.3f}'.format(ap)
        print('AP: %s (%s)' % (ap_str, cl))
    mAP_str = '{0:.3f}'.format(mAP)
    print('mAP: %s\n' % mAP_str)




def single_img_test(img_path, input_size, model, use_cuda=True, use_float16=False):
    # tf bilinear interpolation is different from any other's, just make do
    threshold = 0.05
    iou_threshold = 0.5

    image_name = img_path.replace('\\', '/').split('/')[-1]

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    # display(out, ori_imgs, imshow=False, imwrite=True)

    # print('running speed test...')
    # with torch.no_grad():
    #     print('test1: model inferring and postprocessing')
    #     print('inferring image for 10 times...')
    #     t1 = time.time()
    #     for _ in range(10):
    #         _, regression, classification, anchors = model(x)
    #
    #         out = postprocess(x,
    #                           anchors, regression, classification,
    #                           regressBoxes, clipBoxes,
    #                           threshold, iou_threshold)
    #         out = invert_affine(framed_metas, out)
    #
    #     t2 = time.time()
    #     tact_time = (t2 - t1) / 10
    #     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')
    det_num = len(out[0]['class_ids'])
    det = []
    for i in range(det_num):
        det.append([image_name, out[0]['class_ids'][i], out[0]['scores'][i], tuple(out[0]['rois'][i])])
    return det

if __name__ == '__main__':
    opt = get_args()
    test(opt)