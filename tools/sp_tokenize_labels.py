import os
import sys
import sentencepiece as spm


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} model_file wrd_file1 wrd_file2 ...")
    proc = spm.SentencePieceProcessor(model_file=sys.argv[1])

    for wrd_file in sys.argv[2:]:
        if not wrd_file.endswith(".wrd"):
            print(f"skips {wrd_file}")
            continue
        out_file = wrd_file.replace(".wrd", ".bpe")
        assert not os.path.exists(out_file), f"{out_file} exists"
        print(f"process {wrd_file} to {out_file}")
        with open(wrd_file) as fin, open(out_file, "w") as fout:
            for line in fin:
                tokens = proc.tokenize(line.strip(), out_type=str)
                print(" ".join(tokens), file=fout)


if __name__ == "__main__":
    main()
