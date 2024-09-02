import argparse
import os
import random
import csv
import json
import torch
import time
import numpy as np
import pickle
import warnings
from multiprocessing import Process

from tools.func import *
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image

# cambrian-phi3-3b
# conv_mode = "phi3"

# cambrian-8b
conv_mode = "llama_3" 
# cambrian-34b
# conv_mode = "chatml_direct"

# cambrian-13b
# conv_mode = "vicuna_v1"

OCL_DATA_PATH = '/data/DATA/OCL_DATA/OCL_data/data/resources'
OCL_IMG_PREFIX = '/data/DATA/OCL_DATA/OCL_data/data'
MODEL_PATH = os.path.expanduser("/data/MODEL/Cambrian-1/cambrian-8b")
GPU_MEM_NEED = 23 * 1024

warnings.filterwarnings('ignore')

prompt={
    # sort
    '1': "Analyze this image of the {sam_obj}, find the most likely {kw} from {candidate_list} and rank in descending order of probability. The result needs to be given in the form of a python list.",
    # '2': "Analyze this image of the {sam_obj}, find the most likely {kw} from {candidate_list} and rank in descending order of probability.",
    # '3': "Identify the {sam_obj}'s {kw}, and rank {candidate_list} from the most relevant to the least. The result needs to be given in the form of a python list.",
    '4': "Find the most likely {kw} in this image of the {sam_obj} and rank {candidate_list} in descending order of probability. The result needs to be given in the form of a python list.",
    # top10
    '5': "Analyze this image of the {sam_obj} to provide the top 10 {kw} from {candidate_list}.",
    '6': "Selected from {candidate_list}, list the top 10 {kw} of the object {sam_obj} in this image.",
    '7': "Analyze this image to identify the {kw} of the {sam_obj}. Sort the top10 candidate {kw} from {candidate_list}.",
    '8': "Given the {sam_obj} in this image, list the top 10 most distinguished {kw} identified using {candidate_list}.", 
    # list
    '9': "What's the {kw} of the object {sam_obj} in this image. The result needs to be selected from {candidate_list}.", 
    '10': "Give the {kw} of the object {sam_obj} in this image. The result must be selected from {candidate_list}.", 
    # muti quest
    '11': "Give the {sam_obj} in this image, for the following questions, answer each question with 'yes' or 'no'. {questions}", 
    # causal
    'causal1': "Give the {sam_obj} in this image, for the following question, make a judgment successively.\n{questions}", 
    }
def set_and_parse_args():
    """参数解析"""
    parser = argparse.ArgumentParser(description="运行cambrian")
    
    parser.add_argument('--output', '-o', type = str, required=True, help = "输出结果目录")
    parser.add_argument('--check', '-c', action='store_true', help='仅检查gpu内存情况')
    parser.add_argument('--num', '-n', type = int,default= 0,required=False, help = "样本数, 默认计算所有样本")
    parser.add_argument('--start', '-s', type = int,default= 0,required=False, help = "起始样本index, 默认0")
    parser.add_argument('--process', type = int,default= 1,required=False, help = "拆分到多张显卡并行, 默认不拆分")
    parser.add_argument('--key_word', '-k', type = str,default= 'attr',required=False, help = "attr/aff, 默认attr")
    parser.add_argument('--prompt', '-p', type = str,default= '1',required=False, help = "index or question")
    parser.add_argument('--noshuffle', action='store_false', help='response中未出现条目不进行打乱')
    parser.add_argument('--nomodel', action='store_true', help='不使用模型')
    parser.add_argument('--use_gt', action='store_true', help='将gt置于clearned_response头部')
    parser.add_argument('--candidate_num', type = int,default= 0,required=False, help = "只考虑candidate list前n个")

    args = parser.parse_args()

    return args

args = set_and_parse_args()

def process(image, question, tokenizer, image_processor, model_config, gpu_id=0):
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
    image_tensor = process_images([image], image_processor, model_config,gpu_id=gpu_id)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(gpu_id)

    return input_ids, image_tensor, image_size, prompt

def run_wo_model(output_dir, cand_n, sample_num=0, start=0,key_word='attr', use_gt=False):
    os.makedirs(output_dir, exist_ok=True)
    sample_num = len(val_data) if sample_num <= 0 else sample_num
    with open(os.path.join(output_dir,'clearned_response.csv'), 'w', newline='') as cr, open(os.path.join(output_dir,'prediction.csv'), 'w', newline='') as p, open(os.path.join(output_dir,'gt_attr.csv'), 'w', newline='') as g:
        cr_writer = csv.writer(cr)
        p_writer = csv.writer(p)
        g_writer = csv.writer(g)
        start_time = time.time()
        for sam_index in range(start,start+sample_num):
            sample=val_data[sam_index]
            sam_obj=sample['objects'][0]['obj']
            skw=[i for i in sample['objects'][0][key_word] if i<cand_n]
            print(f"################################\nindex: {sam_index}; obj: {sam_obj}")
            for i in skw:
                print(f"attr: {candidate_list[i]}; ",end="")
            print("\n################################")
            gt = np.zeros(len(candidate_list), dtype=int)
            gt[skw]=1
            g_writer.writerow(gt.tolist())
            if use_gt:
                gt_at_top=list(dict.fromkeys([candidate_list[i] for i in skw]))
                unappear_attribute=list(set(candidate_list)-set(gt_at_top))
                random.shuffle(unappear_attribute)
                clearned_response=gt_at_top+unappear_attribute
            else:
                clearned_response,_=filter_response_to_ids(candidate_list, "", shuffle=args.noshuffle)
            cr_writer.writerow(clearned_response)
            index_response = [clearned_response.index(i) for i in candidate_list]
            p_response=linear(index_response,len(clearned_response)).tolist()
            p_writer.writerow(p_response)
            cr.flush()
            p.flush()
            g.flush()
        end_time = time.time()
    return start_time,end_time,sample_num

def load_model(model_path=MODEL_PATH, gpu_id=0):
    model_name = get_model_name_from_path(model_path)
    device_used="cuda:{}".format(gpu_id)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name,device_map=device_used)
    print(model)
    print("model.config",model.config)
    exit()
    return tokenizer, model, image_processor

def run_model(tokenizer, model, image_processor, output_dir, cand_n, gpu_id=0, sample_num=0, start=0,key_word='attr',kw = "attributes"):
    device_used="cuda:{}".format(gpu_id)
    os.makedirs(output_dir, exist_ok=True)
    sample_num = len(val_data) if sample_num <= 0 else sample_num
    with open(os.path.join(output_dir,'response.txt'), 'w',) as r, open(os.path.join(output_dir,'clearned_response.csv'), 'w', newline='') as cr, open(os.path.join(output_dir,'prediction.csv'), 'w', newline='') as p, open(os.path.join(output_dir,'gt_attr.csv'), 'w', newline='') as g:
        cr_writer = csv.writer(cr)
        p_writer = csv.writer(p)
        g_writer = csv.writer(g)
        start_time = time.time()
        response_num,unappear_num=0,0
        for sam_index in range(start,start+sample_num):
            sample=val_data[sam_index]
            image_path = os.path.join(OCL_IMG_PREFIX,sample['name'])
            sam_obj=sample['objects'][0]['obj']
            skw=[i for i in sample['objects'][0][key_word] if i<cand_n]
            print(f"################################\nindex: {sam_index}; image_path: {image_path}; obj: {sam_obj}")
            for i in skw:
                print(f"attr: {candidate_list[i]}; ",end="")
            print("\n################################")
            gt = np.zeros(len(candidate_list), dtype=int)
            gt[skw]=1
            g_writer.writerow(gt.tolist())
            if args.prompt in prompt:
                if args.prompt in ['11','12']:
                    questions=""
                    idx = 0
                    for k in candidate_list:
                        idx += 1
                        if kw=="attributes":
                            questions += f" IDX{idx}: Is {sam_obj} {k}? "
                            # questions += f"Is {sam_obj} {k}?\n"
                        elif kw=="affordances":
                            questions += f" IDX{idx}: Can {sam_obj} {k}? "
                            # questions += f"Can {sam_obj} {k}?\n"
                    question=prompt[args.prompt].format(sam_obj=sam_obj,questions=questions)
                else:
                    question=prompt[args.prompt].format(sam_obj=sam_obj,kw=kw,candidate_list=candidate_list)
            else:
                question=args.prompt.format(sam_obj=sam_obj,kw=kw,candidate_list=candidate_list)
            print("###Q:",question)
            image = Image.open(image_path).convert('RGB')
            input_ids, image_tensor, image_sizes, _ = process(image, question, tokenizer, image_processor, model.config, gpu_id)
            input_ids = input_ids.to(device=device_used, non_blocking=True)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    num_beams=1,
                    # max_new_tokens=1024,
                    max_new_tokens=512,
                    use_cache=True)

            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print("###A:",response)
            r.write(f"{response}\n")
            clearned_response,stat=filter_response_to_ids(candidate_list, response, shuffle=args.noshuffle)
            print("stat",stat)
            response_num+=stat[2]
            unappear_num+=stat[3]
            cr_writer.writerow(clearned_response)
            index_response = [clearned_response.index(i) for i in candidate_list]
            p_response=linear(index_response,len(clearned_response)).tolist()
            p_writer.writerow(p_response)
            r.flush()
            cr.flush()
            p.flush()
            g.flush()
        end_time = time.time()
    return start_time,end_time,sample_num,response_num,unappear_num

def load_run(output_dir, cand_n, model_path=MODEL_PATH, gpu_id=0, sample_num=0, start=0,key_word='attr',kw = "attributes"):
    torch.cuda.set_device(gpu_id)
    tokenizer, model, image_processor = load_model(model_path, gpu_id)
    start_time,end_time,sample_num,response_num,unappear_num=run_model(tokenizer, model, image_processor, output_dir, cand_n, gpu_id, sample_num, start, key_word, kw)
    print(f"总运行时间：{end_time - start_time:.2f}秒")
    print(f"样本平均运行时间：{(end_time - start_time)/sample_num:.2f}秒")
    print(f"response/candidate平均覆盖率：{100*response_num/(response_num + unappear_num):.2f}%")
    return start_time,end_time,sample_num

if not args.nomodel:
    available_gpus = gpu_mem_check(GPU_MEM_NEED)
    print(f"可用显卡 {len(available_gpus)} 张: {available_gpus}")
    if args.check:
        exit()

    if len(available_gpus) < args.process:
        exit("显卡不足，请减少并行数量或稍后重试！")

print = decorate(print)
temperature = 0
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(os.path.join(OCL_DATA_PATH,"OCL_class_attribute.json"),"r") as f:
    attr_list = json.load(f)

with open(os.path.join(OCL_DATA_PATH,"causal_many_shot300.json"),"r") as f:
    many300 = json.load(f)

aff_list=[]
with open(os.path.join(OCL_DATA_PATH,"OCL_class_affordance_word.txt"), 'r') as f:
    for line in f:
        aff_list.append(line.strip())

with open(os.path.join(OCL_DATA_PATH, 'OCL_annot_test.pkl'), 'rb') as f:
    val_data = pickle.load(f)

kw="attributes" if args.key_word=='attr' else "affordances"
candidate_list = attr_list if args.key_word=='attr' else aff_list
cand_n = len(candidate_list) if args.candidate_num ==0 else args.candidate_num
candidate_list=candidate_list[:cand_n]
if args.nomodel:
    run_wo_model(args.output, cand_n, sample_num=args.num, start=args.start,key_word=args.key_word,use_gt=args.use_gt)
else:
    interv_list=split4fork(args.num,args.process,args.start)
    res2merge = {'clearned_response.csv': [], 'gt_attr.csv': [], 'prediction.csv': [], 'response.txt': []}
    processes = []
    for i,interv in enumerate(interv_list):
        split_output=os.path.join(args.output,f"split_{i}")
        for k in res2merge:
            res2merge[k].append(os.path.join(split_output,k))
        p = Process(target=load_run, args=(split_output,cand_n,), kwargs={'gpu_id': available_gpus[i],'sample_num': interv[1],'start': interv[0],'key_word': args.key_word,'kw': kw})
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    for k in res2merge:
        command = f"cat {' '.join(res2merge[k])} > {os.path.join(args.output,k)}"
        os.system(command)

prediction = np.genfromtxt(os.path.join(args.output,'prediction.csv'), delimiter=',', skip_header=0)
gt_attr = np.genfromtxt(os.path.join(args.output,'gt_attr.csv'), delimiter=',', skip_header=0)
mAP=mAP_evaluator(prediction,gt_attr,os.path.join(args.output,'AP'))
print("########################################\nmAP=",mAP)
