from torch.utils.data import Subset
from transformers import BertTokenizerFast
import torch

# used for teacher allocating
def teacher_allocate(teacher_lst, k):
    n = len(teacher_lst)
    part_size = n // k
    remainder = n % k  # Handle cases where n is not perfectly divisible

    parts = []
    start = 0
    for i in range(k):
        extra = 1 if i < remainder else 0  # Distribute remaining elements
        end = start + part_size + extra
        parts.append(teacher_lst[start:end])
        start = end
    return parts


# divide chunk into slices
def chunk_slice(chunk_set, r):
    sliced_chunk = []
    for chunk in chunk_set:
        slice_set = []
        chunk_size = len(chunk)
        slice_size = chunk_size//r
        sizes = [slice_size] * r
        sizes[-1] += chunk_size % r
        start = 0
        for size in sizes:
            slice = Subset(chunk, range(start, start+size))
            start = start + size
            slice_set.append(slice)
        sliced_chunk.append(slice_set)
    
    return sliced_chunk

def chunk_slice_label(chunk_set, r):
    sliced_chunk = []
    for chunk_tensor in chunk_set:
        chunk = chunk_tensor[0]
        slice_set = []
        chunk_size = len(chunk)
        slice_size = chunk_size//r
        sizes = [slice_size] * r
        sizes[-1] += chunk_size % r
        start = 0
        for size in sizes:
            slice = chunk[start:start+size]
            start = start + size
            slice_set.append(slice)
        sliced_chunk.append(slice_set)
    return sliced_chunk


def tokenize(batch):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

def tokenizesst5(batch):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)


def tokenize_and_get_tesnors(tokenizer, input, device, use_roberta = False):
    if len(input[0]) == 2:
        features = tokenizer([x[0] for x in input], [x[1] for x in input] , max_length=256, truncation=True, padding=True)
    else:
        features = tokenizer([x[0] for x in input], max_length=256, truncation=True,
                             padding=True)

    all_input_ids = torch.tensor([f for f in features["input_ids"]], dtype=torch.long)
    all_attention_mask = torch.tensor([f for f in features["attention_mask"]], dtype=torch.long)
    all_input_ids = all_input_ids.to(device)
    all_attention_mask = all_attention_mask.to(device)

    if not use_roberta:
        all_token_type_ids = torch.tensor([f for f in features["token_type_ids"]], dtype=torch.long)
        all_token_type_ids = all_token_type_ids.to(device)
    else:
        all_token_type_ids = None

    if not use_roberta:
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "token_type_ids": all_token_type_ids}
    else:
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask}

    return inputs
