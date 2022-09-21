import pandas as pd


def read_data(fn):
    data = pd.read_json(fn)

    labels = data.tag.to_list()
    texts = data.sentence.to_list()
    doc_id = data.doc_id.to_lits()
    is_key = data.keysencence.to_list()

    return labels, texts, doc_id, is_key
