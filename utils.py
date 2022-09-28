import pandas as pd


def read_data(fn):
    data = pd.read_json(fn)

    fine_tags = data.fine_tag.to_list()
    coarse_tags = data.coarse_tag.to_list()
    texts = data.sentence.to_list()
    doc_ids = data.doc_id.to_list()

    return texts, fine_tags, coarse_tags, doc_ids


def labels_to_map(fn):
    """
    :param fn: 태그(세부 태그, 대분류 태그)의 vocab 파일 (.txt)
    :return: 정수_to_tag 딕셔너리
    """
    vocab_map = {}
    with open(fn, 'r', encoding='utf-8') as vocab:
        for label in vocab:
            label = label.strip()
            vocab_map[len(vocab_map)] = label

    return vocab_map
