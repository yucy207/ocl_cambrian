import argparse
import os
import json
import random
import re
import csv
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import pickle
import sklearn.metrics as sklmetric

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

os.environ['HF_DATASETS_OFFLINE']="1"
os.environ['HF_HUB_OFFLINE']="1"
os.environ['TRANSFORMERS_OFFLINE']="1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
# cambrian-phi3-3b
# conv_mode = "phi3"

# cambrian-8b
conv_mode = "llama_3" 
gpu_id=2
device_used="cuda:{}".format(gpu_id)
torch.cuda.set_device(gpu_id)
# cambrian-34b
# conv_mode = "chatml_direct"

# cambrian-13b
# conv_mode = "vicuna_v1"

def process(image, question, tokenizer, image_processor, model_config):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(gpu_id)

    return input_ids, image_tensor, image_size, prompt

import torch
import numpy as np
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = os.path.expanduser("/data/MODEL/Cambrian-1/cambrian-8b")
model_name = get_model_name_from_path(model_path)
# print(model_path)
# print(model_name)
# torch.cuda.empty_cache()
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,device_map=device_used)

def decorate(func):
    def flush_print(*args, **kwargs):
        return func(*args, **kwargs, flush=True)
    return flush_print
print = decorate(print)

temperature = 0
def average_precision(truth, scores):
    if np.sum(truth > 0) > 0:
        # AUC sklmetric.roc_auc_score(truth, scores)
        a = sklmetric.average_precision_score(truth, scores)
        assert not np.isnan(a)
        return a
    else:
        return np.nan

def mAP_evaluator(prediction, gt_attr, store_ap=None, return_vec=False):
    """prediction, gt_attr: (#instance, #category)
    return mAP(float)"""
    assert prediction.shape == gt_attr.shape

    assert not np.any(np.isnan(prediction)), str(np.sum(np.isnan(prediction)))
    assert not np.any(np.isnan(gt_attr)), str(np.sum(np.isnan(gt_attr)))

    ap = np.zeros((gt_attr.shape[1],))
    pos = np.zeros((gt_attr.shape[1],))  # num of positive sample

    for dim in range(gt_attr.shape[1]):
        # rescale ground truth to [-1, 1]

        gt = gt_attr[:, dim]
        mask = (gt >= 0)

        gt = 2 * gt[mask] - 1  # = 0.5 threshold
        est = prediction[mask, dim]

        ap[dim] = average_precision(gt, est)
        pos[dim] = np.sum(gt > 0)

    if store_ap is not None:
        import os
        assert not os.path.exists(store_ap + '.txt')
        with open(store_ap + '.txt', 'w') as f:
            for dim in range(gt_attr.shape[1]):
                f.write("Dim %d AP %f\n" % (dim, ap[dim]))

    if return_vec:
        return ap
    else:
        mAP = np.nanmean(ap)
        return mAP * 100

def load_class_json(name):
    with open(os.path.join("/data/DATA/OCL_DATA/OCL_data/data/resources",f"OCL_class_{name}.json"),"r") as fp:
        return json.load(fp)
    
attr_list = load_class_json("attribute")
print(len(attr_list))

with open('/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_annot_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

def filter_response_to_ids(cand_list, response):

    # 使用正则表达式提取单引号内的内容
    words = re.findall(r"'(.*?)'", response)
    # response存在重复的情况
    words=list(dict.fromkeys(words))
    clearned_response = []
    for word in words:
        if word in cand_list:
            clearned_response.append(word)
    if len(clearned_response) < len(cand_list):
        unappear_attribute = list(set(cand_list)-set(clearned_response))
        random.shuffle(unappear_attribute)
        clearned_response += unappear_attribute
    return clearned_response

def linear(x, k=200.):
    x = torch.tensor(x, dtype=torch.float32)
    return 1. - 1. / k * x

prediction = []
gt_attr = []
output_dir="/home/lihong/yuchenyang/cambrian/OCL_output/"
with open(output_dir+'response.txt', 'w',) as r, open(output_dir+'clearned_response.csv', 'w', newline='') as cr, open(output_dir+'prediction.csv', 'w', newline='') as p, open(output_dir+'gt_attr.csv', 'w', newline='') as g:
    cr_writer = csv.writer(cr)
    p_writer = csv.writer(p)
    g_writer = csv.writer(g)
    # while True:
        # image_path = input("image path: ")
        # question = input("question: ")
    # while True:
        # sam_index = input("sample index: ")
        # sample=val_data[int(sam_index)]
    for sam_index in range(len(val_data)):
        sample=val_data[sam_index]
        image_path = os.path.join("/data/DATA/OCL_DATA/OCL_data/data",sample['name'])
        sam_obj=sample['objects'][0]['obj']
        print(f"################################\nindex: {sam_index}; image_path: {image_path}; obj: {sam_obj}")
        for attr in sample['objects'][0]['attr']:
            print(f"attr: {attr_list[attr]}; ",end="")
        print("\n################################")
        gt = np.zeros(len(attr_list), dtype=int)
        gt[sample['objects'][0]['attr']]=1
        g_writer.writerow(gt.tolist())
        gt_attr.append(gt.tolist())
        question = f"Find the most likely attributes in this image of the {sam_obj} and rank {attr_list} in descending order of probability. The result needs to be given in the form of a list."
        # question1 = input("question1: ")The result needs to be given in the form of a list (must be the prototype in the list).
        # question += question1
        image = Image.open(image_path).convert('RGB')
        input_ids, image_tensor, image_sizes, prompt = process(image, question, tokenizer, image_processor, model.config)
        # input_ids = input_ids.to(device='cuda', non_blocking=True)
        input_ids = input_ids.to(device=device_used, non_blocking=True)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                num_beams=1,
                # max_new_tokens=128,
                max_new_tokens=512,
                use_cache=True)

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(response)
        r.write(f"{response}\n")
        clearned_response=filter_response_to_ids(attr_list, response)
        cr_writer.writerow(clearned_response)
        index_response = [attr_list.index(x) for x in clearned_response]
        p_response=linear(index_response,len(index_response)).tolist()
        p_writer.writerow(p_response)
        prediction.append(p_response)
        r.flush()
        cr.flush()
        p.flush()
        g.flush()
    
prediction=np.array(prediction)
gt_attr=np.array(gt_attr)
mAP=mAP_evaluator(prediction,gt_attr,output_dir+"AP")
print("########################################\nmAP=",mAP)
