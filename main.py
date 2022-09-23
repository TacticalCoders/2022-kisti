import transformers
from transformers import AutoTokenizer


def main():
    vocab_file = "./vocab_kisti.txt"

    tokenizer = AutoTokenizer.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=False,
        tokenizer_type="Mecab")

    model_path = 'model/pytorch_model.bin'
    config = transformers.BertConfig.from_pretrained('model/bert_config_kisti.json')
    model = transformers.BertForPreTraining.from_pretrained(model_path, config=config)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')