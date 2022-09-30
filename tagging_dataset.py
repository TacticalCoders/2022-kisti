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

        encoding = self.process_texts_for_bert(texts)

        return_value = {
            'input_ids': encoding['total_input_ids'],  # max_length 만큼 잘라진 길이여야 함...
            'attention_mask': encoding['total_attention_mask'],  # attention mask도 패딩을 제외한
            'coarse_tags': torch.tensor(coarse_tags, dtype=torch.long),
            'fine_tags': torch.tensor(fine_tags, dtype=torch.long)
        }
        if self.with_text:
            return_value['texts'] = texts

        if self.with_doc_id:
            return_value['doc_ids'] = doc_ids

        return return_value

    def process_texts_for_bert(self, texts):
        """
        :param texts: 데이터셋의 원본 텍스트 문장
        :return: 딕셔너리, 정수로 변환된 input_ids, attention_mask 를 torch.tensor 형태(dtype=torch.long)로 반환
        """
        max_length = self.max_length
        total_input_ids = []
        total_attention_mask = []

        tokenized_texts = map(self.tokenizer.tokenize, texts)
        for tokens in tokenized_texts:
            tokens = ["[CLS]"] + tokens
            tokens = tokens[:max_length - 1]
            tokens.append("[SEP]")

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            assert len(input_ids) <= max_length

            attention_mask = [1] * len(input_ids)
            padding = [0] * (max_length - len(input_ids))

            input_ids += padding
            attention_mask += padding

            total_input_ids.append(input_ids)
            total_attention_mask.append(attention_mask)

        total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
        total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)

        return {
            'total_input_ids': total_input_ids,
            'total_attention_mask': total_attention_mask
        }


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
