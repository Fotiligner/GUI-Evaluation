import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# from peft import AutoPeftModelForCausalLM
import ast
import json
import re
import argparse
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from process_utils import pred_2_point, extract_bbox
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
from qwen_vl_utils import process_vision_info
import base64

from typing import Dict, Any, List, Union, TypedDict, Optional

class ContentItem(TypedDict, total=False):
    type: str
    text: Optional[str]
    image_url: Optional[str]

class Message(TypedDict):
    role: str
    content: Union[str, List[ContentItem]]



logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--qwen_path', type=str, required=False)
parser.add_argument('--lora_path', type=str, required=False)
parser.add_argument('--screenspot_imgs', type=str, required=False)
parser.add_argument('--screenspot_test', type=str, required=False)
parser.add_argument('--task', type=str, required=True)
args = parser.parse_args()

args.screenspot_imgs = "/home/hadoop-ba-dealrank/dolphinfs_hdd_hadoop-ba-dealrank/libingqian/ScreenSpot_v2/screenspotv2_image"
args.screenspot_test = "/home/hadoop-ba-dealrank/dolphinfs_hdd_hadoop-ba-dealrank/libingqian/ScreenSpot_v2/screenspotv2_image"

model_path = "/home/hadoop-ba-dealrank/dolphinfs_hdd_hadoop-ba-dealrank/libingqian/Qwen25_vl_3b/Qwen/Qwen2.5-VL-3B-Instruct"

torch_dtype = torch.bfloat16
attn_implementation = "flash_attention_2"
device_map ="auto"


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation=attn_implementation).eval()
# model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_path)

image_path = "/home/hadoop-ba-dealrank/dolphinfs_hdd_hadoop-ba-dealrank/libingqian/ScreenSpot/web_f680f8e6-e068-4ca9-9be6-2f78c2899e0a.png"




if args.task == "all":
    tasks = ["mobile", "desktop", "web"]
else:
    tasks = [args.task]
tasks_result = []
result = []
for task in tasks:
    dataset = "screenspot_" + task + "_v2.json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print("Num of sample: " + str(len(screenspot_data)))
    prompt_origin = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)? The output position should follow the format: `(x,y)`. (0,0) is the top-left corner of the screen."
    prompt_origin_qwen = "Generate the bounding box of {}"
    prompt_test = "Please describe the image."
    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0
    for j, item in tqdm(enumerate(screenspot_data)):
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        image = Image.open(img_path)
        instruction = item["instruction"]
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img_size = image.size

        print(img_size)
        # print(img_size)
        # bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(img_path),
                    },
                    {"type": "text", "text": prompt_origin.format(instruction)},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

        response = output_text[0]

        # print(response)

        try:
            if 'box' in response:
                pred_bbox = extract_bbox(response)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                click_point = [item / 1000 for item in click_point]
            else:
                click_point = pred_2_point(response)
            # click_point[0] = click_point[0] / img_size[0]
            # click_point[1] = click_point[1] / img_size[1]
            print(bbox)
            print(click_point)
            point = click_point
            # draw = ImageDraw.Draw(image)
            # draw.rectangle(bbox, outline="red", width=2)  # 绘制 BBox（红色框）
            # draw.ellipse([point[0]-30, point[1]-30, point[0]+30, point[1]+30], fill="blue")  # 绘制点（蓝色圆点）

            # # 4. 使用 Matplotlib 显示图片
            # plt.imshow(image)
            # plt.axis("off")  # 不显示坐标轴
            # plt.savefig("output.png", bbox_inches="tight", pad_inches=0, dpi=300)

            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                logging.info("match " + str(corr_action / num_action))
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("unmatch " + str(corr_action / num_action))
            result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "pred": click_point,
                           "type": item["data_type"], "source": item["data_source"]})
        except:
            num_wrong_format += 1
            if item["data_type"] == 'text':
                text_correct.append(0)
            else:
                icon_correct.append(0)
            logging.info("Step: " + str(j) + " wrong format")

    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
    logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

    text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
    icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
    tasks_result.append([text_acc, icon_acc])

logging.info(tasks_result)

