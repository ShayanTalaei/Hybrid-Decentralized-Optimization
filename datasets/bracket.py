from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines

import torch
import torch.utils.data
import numpy as np


class BracketTokenizer:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", bos_token="<start>", eos_token="<stop>"):
        self.vocab = [bos_token, eos_token, pad_token, unk_token] + ['(', ')']
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        self.pad_token_id = self.word_to_index[pad_token]
        self.unk_token_id = self.word_to_index[unk_token]
        self.bos_token_id = self.word_to_index[bos_token]
        self.eos_token_id = self.word_to_index[eos_token]

    def encode(self, text, max_length=None):
        """This method takes a natural text and encodes it into a sequence of token ids using the vocabulary.

        Args:
            text (str): Text to encode.
            max_length (int, optional): Maximum encoding length. Defaults to None.

        Returns:
            List[int]: List of token ids.
        """
        token_ids = [self.bos_token_id] + [self.word_to_index.get(word, self.unk_token_id) for word in text.split()] + [
            self.eos_token_id]
        if max_length:
            token_ids = token_ids[:max_length + 2] + [self.pad_token_id] * (max_length - len(token_ids) + 2)
        return token_ids

    def decode(self, sequence, skip_special_tokens=True):
        """This method takes a sequence of token ids and decodes it into a language tokens.

        Args:
            sequence (List[int]): Sequence to be decoded.
            skip_special_tokens (bool, optional): Whether to skip special tokens when decoding. Defaults to True.

        Returns:
            List[str]: List of decoded tokens.
        """
        tokens = [self.index_to_word[idx] for idx in sequence.cpu().detach().numpy()]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]]
        return "".join(tokens)


class BracketDataset(Dataset):
    """
    SADataset in Pytorch
    """

    def __init__(self, data_repo, tokenizer, sent_max_length=512):
        self.tokenizer = tokenizer

        self.pad_token = self.tokenizer.pad_token
        self.pad_id = self.tokenizer.pad_token_id
        labels = ['correct', 'incorrect']

        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.id_to_label = {i: label for i, label in enumerate(labels)}

        self.text_samples = []
        self.samples = []

        print("Building Dataset...")

        with jsonlines.open(data_repo, "r") as reader:
            for sample in tqdm(reader.iter()):
                # print(sample)
                self.text_samples.append(sample)
                input_ids = self.tokenizer.encode(sample['input'], max_length=sent_max_length)
                # self.samples.append({"ids": input_ids[:-1], "label": input_ids[1:]})
                self.samples.append({"ids": torch.tensor(input_ids), "label": torch.tensor(self.label_to_id[sample['label']])})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # cp = deepcopy(self.samples[index])
        # return cp['ids'], cp['label']
        return self.samples[index]['ids'], self.samples[index]['label']
    def __padding(self, inputs, max_length=-1):
        """
        Pad inputs to the max_length.

        INPUT: 
          - inputs: input token ids
          - max_length: the maximum length you should add padding to.

        OUTPUT: 
          - pad_inputs: token ids padded to `max_length` """

        if max_length < 0:
            max_length = np.max([len(input_ids) for input_ids in inputs])

        pad_inputs = [input_ids + [self.pad_id] * (max_length - len(input_ids)) for input_ids in inputs]

        return pad_inputs

    def collate_fn(self, batch):
        """
        Convert batch inputs to tensor of batch_ids and labels.

        INPUT: 
          - batch: batch input, with format List[Dict1{"ids":..., "label":...}, Dict2{...}, ..., DictN{...}]

        OUTPUT: 
          - tensor_batch_ids: torch tensor of token ids of a batch, with format Tensor(List[ids1, ids2, ..., idsN])
          - tensor_labels: torch tensor for corresponding labels, with format Tensor(List[label1, label2, ..., labelN])
        """

        tensor_batch_ids = torch.tensor([sample["ids"] for sample in batch])
        tensor_labels = torch.tensor([sample["label"] for sample in batch])

        return tensor_batch_ids, tensor_labels
    