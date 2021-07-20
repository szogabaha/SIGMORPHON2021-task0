import os

import fire

LANGS = [
    'afb', 'ame', 'ara', 'aym', 'bul', 'ckb', 'cni', 'evn',
    'heb', 'itl', 'kod', 'lud', 'nld', 'pol', 'rus', 'see',
    'syc', 'vep', 'ail', 'amh', 'arz', 'bra', 'ces', 'ckt',
    'deu', 'gup', 'ind', 'kmr', 'krl', 'mag', 'olo', 'por',
    'sah', 'spa', 'tyv',
]

MAX_HALL = 10_000
IN_DIR = "part1/original"
OUT_DIR = "part1/processed"


def read_file(file):
    with open(file) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            toks = line.split("\t")
            if len(toks) == 3:
                yield toks


def regular():
    for lang in sorted(LANGS):
        for mode in ["dev"]:
            with open(f"{OUT_DIR}/{lang}.{mode}", "w") as fp:
                for toks in read_file(f"{IN_DIR}/{lang}.{mode}"):
                    print(*toks, sep="\t", file=fp)


def halluication():
    for lang in sorted(LANGS):
        mode = "train"
        if not os.path.isfile(f"{IN_DIR}/{lang}.hall"):
            print("missing .hall for", lang)
            continue
        with open(f"{OUT_DIR}/{lang}.{mode}", "w") as fp:
            for toks in read_file(f"{IN_DIR}/{lang}.{mode}"):
                print(*toks, sep="\t", file=fp)
            for i, toks in enumerate(read_file(f"{IN_DIR}/{lang}.hall")):
                if i == MAX_HALL:
                    break
                print(toks[0], toks[1], f"fake;{toks[2]}", sep="\t", file=fp)


class Main:
    def regular(self):
        regular()

    def hall(self):
        halluication()

    def all(self):
        self.regular()
        self.hall()

    def gen_langs(self):
        langs = []
        for family in LANGS.keys():
            langs.extend(LANGS[family])
        print(" ".join(sorted(langs)))
        print(len(langs))

    def gen_family(self):
        family = list(LANGS.keys())
        print(" ".join(sorted(family)))
        print(len(family))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fire.Fire(Main)
