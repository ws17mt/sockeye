import sys
import sockeye.vocab


def main():
    infiles = sys.argv[1:-1]
    outfile = sys.argv[-1]

    voc = sockeye.vocab.build_from_path(infiles)
    sockeye.vocab.vocab_to_json(voc, outfile)


if __name__ == "__main__":
    main()
