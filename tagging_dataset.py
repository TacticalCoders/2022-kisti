import torch
from torch.utils.data import Dataset


class SentenceTaggingCollator:
    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,  # max_length를 넘어가면 잘라냄
            return_tensor="pt",
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class SentenceTaggingDataset(Dataset):

    def __init__(self, texts, labels, doc_id, is_key):
        self.texts = texts
        self.labels = labels
        self.doc_id = doc_id
        self.is_key = is_key

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        doc_id = self.doc_id[item]
        is_key = self.is_key[item]

        return {
            'text': text,
            'label': label,
            'doc_id': doc_id,
            'is_key': is_key
        }
