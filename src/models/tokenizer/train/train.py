from datasets import load_dataset
from ...ocr_model.model.TexTeller import TexTeller
from ...globals import VOCAB_SIZE


if __name__ == '__main__':
    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-raw')
    dataset = load_dataset("/home/lhy/code/TeXify/src/models/ocr_model/train/dataset/latex-formulas/latex-formulas.py", "cleaned_formulas")['train']
    new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=dataset['latex_formula'], vocab_size=VOCAB_SIZE)
    new_tokenizer.save_pretrained('/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550Kformulas')
    pause = 1
