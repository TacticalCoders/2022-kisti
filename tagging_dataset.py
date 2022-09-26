import torch
from torch.utils.data import Dataset


class SentenceTaggingCollator:
    def __init__(self, tokenizer, max_length,
                 with_text=True, with_doc_id=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        self.with_doc_id = with_doc_id

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        coarse_tags = [s['coarse_tag'] for s in samples]
        fine_tags = [s['fine_tags'] for s in samples]
        doc_ids = [s['doc_id'] for s in samples]

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
            'coarse_tags': torch.tensor(coarse_tags, dtype=torch.long),
            'fine_tags': torch.tensor(fine_tags, dtype=torch.long)
        }
        if self.with_text:
            return_value['texts'] = texts

        if self.with_doc_id:
            return_value['doc_ids'] = doc_ids

        return return_value


class SentenceTaggingDataset(Dataset):

    def __init__(self, texts, coarse_tags, fine_tags, doc_ids):
        self.texts = texts
        self.coarse_tags = coarse_tags
        self.fine_tags = fine_tags
        self.doc_ids = doc_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        coarse_tag = self.coarse_tags[item]
        fine_tag = self.fine_tags[item]
        doc_id = self.doc_ids[item]

        return {
            'text': text,
            'coarse_tag': coarse_tag,
            'fine_tag': fine_tag,
            'doc_id': doc_id,
        }
