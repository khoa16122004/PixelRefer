from train import LazySupervisedDataset, DataArguments
import json

# ============= Tokenizer setup
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_args.model_path,
#     model_max_length=training_args.model_max_length,
#     padding_side="right",
#     use_fast=True,
# )

data_args = DataArguments
data_args.data_folder = 'videorefer/data/video'
data_args.is_multimodal = True


dataset = LazySupervisedDataset(
    data_path='videorefer/data/marine.json',
    tokenizer=None,
    data_args=data_args
)

print(dataset[0])

