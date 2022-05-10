import fire
import re
from tokenizers import Tokenizer
from tqdm.auto import tqdm


def convert_int_to_chr(line, chr_shift=33):
    if isinstance(line, str):
        line = line.strip().split()
    return "".join([chr(int(c) + chr_shift) for c in line])


def merge_duplicates(line):
    return re.sub(r"(.)\1+", r"\1", line, 0, re.MULTILINE)


def convert_file(input_file, output_file, dedup=True, chr_shift=33, downsample=1):
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in tqdm(fin, desc="converting"):
            line = convert_int_to_chr(line, chr_shift)
            if downsample > 1:
                line = line[::downsample]
            if dedup:
                line = merge_duplicates(line)
            print(line, file=fout)


def train_tokenizer(input_file, output_file, vocab_size=30000, token_type="bpe"):
    from tokenizers import Tokenizer

    if token_type == "bpe":
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        # trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    else:
        raise NotImplementedError(f"token_type={token_type}")

    tokenizer.train([input_file], trainer)
    tokenizer.save(output_file)


def tokenize(tokenizer_file, input_file, output_file):
    tokenizer = Tokenizer.from_file(tokenizer_file)
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip()
            token_ids = tokenizer.encode(line).ids
            print(" ".join([f"{i}" for i in token_ids]), file=fout)


def dedup(input_file, output_file):
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip().split()
            line = [
                line[i] for i in range(len(line)) if i == 0 or line[i] != line[i - 1]
            ]
            print(" ".join(line), file=fout)


if __name__ == "__main__":
    fire.Fire(
        {
            "convert": convert_file,
            "train-tokenizer": train_tokenizer,
            "tokenize": tokenize,
            "dedup": dedup,
        }
    )
