import random

import transformers
from torch.utils.data import DataLoader

import tokenization_kisti as tokenization
import utils
from tagging_dataset import SentenceTaggingDataset, SentenceTaggingCollator


def get_loaders(config, tokenizer,  valid_ratio=.2):
    """
    훈련을 위한 데이터 로더, 검증을 위한 데이터 로더를 반환해주는 함수
    :param fn: 원본 데이터 파일 경로
    :param tokenizer: kisti 토크나이저
    :param config: 설정파일(batch_size와 max_length 필요)
    :param valid_ratio: valid 데이터 비율. 기본 20%
    :return: train Dataloader 클래스, valid Dataloader 클래스
    """

    texts, fine_tags, coarse_tags, doc_ids = utils.read_data(config['train_data_path'])

    print(config['fine_map'])
    print(config['coarse_map'])
    fine_vocab_map = config['fine_map']
    coarse_vocab_map = config['coarse_map']

    fine_tags = list(map(fine_vocab_map.get, fine_tags))  # 텍스트로 된 태그들을 정수로 변환
    coarse_tags = list(map(coarse_vocab_map.get, coarse_tags))

    shuffled = list(zip(texts, fine_tags, coarse_tags, doc_ids))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    fine_tags = [e[1] for e in shuffled]
    coarse_tags = [e[2] for e in shuffled]
    doc_ids = [e[3] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))  # valid_ratio 만큼 전체 데이터셋 분리

    print("texts:", texts[:5])
    print("fine_tags:", fine_tags[:5])
    print("coarse_tags:", coarse_tags[:5])
    print("doc_ids:", doc_ids[:5])

    temp = SentenceTaggingCollator(tokenizer, config['max_length'])
    encoding = temp.process_texts_for_bert(texts[:5])
    print(encoding['total_input_ids'])
    print(encoding['total_attention_mask'])

    tokens = tokenizer.tokenize(texts[0])
    encoded_tokens = tokenizer.convert_tokens_to_ids(tokens)
    print(tokens)
    print(encoded_tokens)

    train_loader = DataLoader(
        # texts, coarse_tags, fine_tags, doc_ids 순서
        SentenceTaggingDataset(texts[:idx], coarse_tags[: idx], fine_tags[:idx], doc_ids[:idx]),
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=SentenceTaggingCollator(tokenizer, config['max_length']),
    )

    valid_loader = DataLoader(
        # texts, coarse_tags, fine_tags, doc_ids 순서
        SentenceTaggingDataset(texts[idx:], coarse_tags[idx:], fine_tags[idx:], doc_ids[idx:]),
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=SentenceTaggingCollator(tokenizer, config['max_length']),
    )

    return train_loader, valid_loader


def main():

    fine_vocab_map = utils.labels_to_ids_vocab("./data/fine_vocab.txt")
    coarse_vocab_map = utils.labels_to_ids_vocab("./data/coarse_vocab.txt")

    config = {
        "mode": "train",
        "train_data_path": "./data/processed_tagging_dataset.json",
        "test_data_path": "",
        "cache_dir_path": "",
        "model_dir_path": "",
        "checkpoint": 0,
        "n_epochs": 5,
        "learning_rate": 1e-5,
        "dropout_rate": 0.3,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "batch_size": 32,
        "test_batch_size": 12,
        "max_length": 512,
        "lstm_hidden": 256,
        "lstm_num_layer": 1,
        "bidirectional_flag": True,
        "fine_tag": 9,
        "coarse_tag": 3,
        "fine_map": fine_vocab_map,
        "coarse_map": coarse_vocab_map,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
    }

    vocab_file = "./vocab_kisti.txt"

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=False,
        tokenizer_type="Mecab")

    train_loader, valid_loader = get_loaders(
        config,
        tokenizer,
        valid_ratio=0.2
    )

    print(
        '|train| =', len(train_loader) * config['batch_size'],
        '|valid| =', len(valid_loader) * config['batch_size'],
    )

    n_total_iterations = len(train_loader) * config['n_epochs']
    print('#total_iters =', n_total_iterations)

    model_path = 'model/pytorch_model.bin'
    bert_config = transformers.BertConfig.from_pretrained('model/bert_config_kisti.json')
    model = transformers.BertForPreTraining.from_pretrained(model_path, config=bert_config)


if __name__ == '__main__':
    main()
