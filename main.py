import tokenization as tokenization
def main(config):

    vocab_file = "./vocab_kisti.txt"

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=False,
        tokenizer_type="Mecab"
)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
