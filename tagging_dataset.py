import torch
from torch.utils.data import Dataset


class SentenceTaggingCollator:
    def __init__(self, tokenizer, max_length,
                 with_text=True, with_doc_id=True, with_is_key=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        self.with_doc_id = with_doc_id
        self.with_is_key = with_is_key

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        doc_ids = [s['doc_id'] for s in samples]
        is_keys = [s['is_key'] for s in samples]

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

        if self.with_doc_id:
            return_value['doc_id'] = doc_ids

        if self.with_is_key:
            return_value['is_key'] = is_keys

        return return_value


class SentenceTaggingDataset(Dataset):

    def __init__(self, texts, labels, doc_ids, is_keys):
        self.texts = texts
        self.labels = labels
        self.doc_ids = doc_ids
        self.is_keys = is_keys

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        doc_id = self.doc_ids[item]
        is_key = self.is_keys[item]

        return {
            'text': text,
            'label': label,
            'doc_id': doc_id,
            'is_key': is_key
        }
