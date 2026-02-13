from train import LazySupervisedDataset, DataArguments, ModelArguments, TrainingArguments
import json
import transformers
import torch

# ============= Model setup
model_args = ModelArguments()
model_args.model_path = 'Qwen/Qwen2-7B-Instruct' # Use Qwen as requested

# ============= Training setup
training_args = TrainingArguments(output_dir="./tmp")
training_args.model_max_length = 512

# ============= Tokenizer setup
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_path,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# SIMULATE: Add time tokens to tokenizer (as done in train.py)
from videorefer.constants import NUM_TIME_BINS
num_time_tokens = NUM_TIME_BINS
time_tokens = [f"<time_{i}>" for i in range(num_time_tokens)]
tokenizer.add_tokens(time_tokens)
print(f"Added {num_time_tokens} time tokens to tokenizer.")

# ============= Data Args setup
data_args = DataArguments()
data_args.data_folder = 'videorefer/data/video' # USE ABSOLUTE PATH
data_args.is_multimodal = True
data_args.image_aspect_ratio = 'pad'
data_args.is_pretraining = False


# Load video processor (lightweight, just config/processor)
data_args.video_processor = transformers.SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

dataset = LazySupervisedDataset(
    data_path='videorefer/data/marine.json', # USE ABSOLUTE PATH
    tokenizer=tokenizer,
    data_args=data_args
)

print(f"Dataset length: {len(dataset)}")
sample = dataset[1]
print("Keys in sample:", sample.keys())

# Decode the input_ids to check if time tokens are present
input_ids = sample['input_ids']
print("\n--- Input IDs ---")
print(input_ids)

label_ids = sample['labels']
print("\n--- Label IDs ---")
print(label_ids)


# Filter out negative IDs (e.g. -200 for image, -201 for video) which cause OverflowError in decode
valid_input_ids = input_ids[input_ids >= 0]
decoded_text = tokenizer.decode(valid_input_ids, skip_special_tokens=False)
print("\n--- Decoded Sample ---")
print(decoded_text)
print("----------------------")

# Verify if any time token exists in decoded text
if "<time_" in decoded_text:
    print("SUCCESS: Time tokens found in the decoded output!")
else:
    print("WARNING: Time tokens NOT found. Check injection logic or duration.")
