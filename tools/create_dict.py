from collections import Counter
import fire


def main(input, output, thres=0):
    counter = Counter()
    with open(input) as f:
        for line in f:
            counter.update(line.strip().split())

    with open(output, "w") as f:
        for tok, count in counter.most_common():
            if count >= thres:
                print(tok, count, file=f)


if __name__ == "__main__":
    fire.Fire(main)
