import pandas as pd


def read_data(fn):
    data = pd.read_json(fn)

    fine_tag = data.fine_tag.to_list()
    coarse_tag = data.coarse_tag.to_list()
    texts = data.sentence.to_list()
    doc_id = data.doc_id.to_lits()

    return fine_tag, coarse_tag, texts, doc_id
