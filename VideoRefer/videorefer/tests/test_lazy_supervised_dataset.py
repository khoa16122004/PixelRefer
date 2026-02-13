import json
from types import SimpleNamespace
import torch

from videorefer.train import LazySupervisedDataset, DataArguments


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.model_max_length = 512

    def __call__(self, text, add_special_tokens=False):
        # simple deterministic token ids per word
        if not text:
            ids = [0]
        else:
            ids = [len(t) % 100 for t in str(text).split()]
        return SimpleNamespace(input_ids=ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # messages: list of {'role':..., 'content':...}
        return " ".join([m["content"] for m in messages])


def test_lazy_supervised_dataset_tokenize_and_shapes(tmp_path):
    # prepare a minimal dataset (no image/video keys)
    sample = {
        "conversations": [
            {"from": "human", "value": "Hello there"},
            {"from": "gpt", "value": "General Kenobi"},
        ]
    }
    data = [sample]
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(data))

    tokenizer = DummyTokenizer()
    data_args = DataArguments()

    ds = LazySupervisedDataset(str(data_file), tokenizer=tokenizer, data_args=data_args)

    item = ds[0]

    # basic expectations
    assert "input_ids" in item and "labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert "masks" in item and isinstance(item["masks"], torch.Tensor)
    assert "ann_indices" in item
