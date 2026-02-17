
import os
import argparse
import json
import torch
import sys
sys.path.append(os.getcwd())  # Ensure we can import videorefer

from videorefer import model_init, mm_infer
from videorefer.mm_utils import process_video
from videorefer.utils import disable_torch_init
from videorefer.data_utils import timestamp_to_time_token
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def run_inference(args):
    disable_torch_init()
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model, processor, tokenizer = model_init(args.model_path)
    
    # Initialize Model components
    for m in model.modules():
        m.tokenizer = tokenizer
    
    model = model.to(device='cuda', dtype=torch.float16)
    
    # Load Data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data_list = json.load(f)
    
    # Select a subset for testing if requested
    if args.num_examples > 0:
        data_list = data_list[:args.num_examples]
    
    output_data = []
    
    print(f"Running inference on {len(data_list)} examples...")
    for idx, item in tqdm(enumerate(data_list)):
        video_name = item['video']
        video_path = os.path.join(args.video_folder, video_name)
        
        # Determine Prompt
        # mm_infer automatically prepends the modal token (<video>)
        question = "Describe the video in detail with timestamps."
            
        # Ignore annotations as per train.py (commented out logic)
        try:
             video_tensor, frame_data, height, width, duration = process_video(
                video_path, 
                processor=processor, 
                aspect_ratio='square', 
                num_frames=16, 
                frame_idx=None
            )
        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
            continue

        # Set masks and indices to defaults used in training
        masks = None 
        ann_indices = [[0]]
        frame_nums = [1]

        # Run Inference
        with torch.inference_mode():
            output = mm_infer(
                video_tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                masks=masks, 
                frame=frame_data,
                ann_indices=[ann_indices], # Batch dim wrap
                frame_nums=frame_nums,
            )
        
        # Get Ground Truth (first assistant response)
        # Replicate training logic to inject time tokens
        timestamps = item.get('timestamp', [])
        conversations = item.get('conversation', [])
        # Get Ground Truth (first assistant response)
        # Replicate training logic to inject time tokens and concatenate all events
        gt_parts = []
        timestamps = item.get('timestamp', [])
        conversations = item.get('conversation', [])
        
        # We assume one-to-one mapping
        for i, conv in enumerate(conversations):
            # Check if this conversation turn has a corresponding timestamp
            current_timestamp = timestamps[i] if i < len(timestamps) else None
            
            for msg in conv:
                 if msg.get('from') in ['gpt', 'Gemini', 'Qwen']:
                     raw_value = msg.get('value', '')
                     if current_timestamp:
                         # Calculate time tokens
                         t_tokens = timestamp_to_time_token(current_timestamp[0], current_timestamp[1], duration)
                         time_str = "".join(t_tokens)
                         gt_parts.append(f"{time_str} {raw_value}")
                     else:
                         gt_parts.append(raw_value)
                     break
        
        gt = " ".join(gt_parts)
        
        print(f"\n[Example {idx}] Video: {video_name}")
        print(f"Prompt: {question}")
        print(f"Prediction: {output}")
        # print(f"Ground Truth: {gt[:100]}...") # Print start of GT
        
        output_data.append({
            'video': video_name,
            'question': question,
            'prediction': output,
            'ground_truth': gt
        })
    
    # Save Results
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--video-folder', type=str, required=True)
    parser.add_argument('--output-file', type=str, default='train_inference_results.json')
    parser.add_argument('--num-examples', type=int, default=5)
    
    args = parser.parse_args()
    run_inference(args)
