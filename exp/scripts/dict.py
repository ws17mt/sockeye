import sys
import collections

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

def threshold_vocab(fname, threshold):
    word_counts = collections.Counter()
    with open(fname) as fin:
        for line in fin:
            for token in line.split():
                word_counts[token] += 1

    ok = set()
    non_ok = set()
    for word, count in sorted(word_counts.items()):
        if count >= threshold or word.find("@L_") == 0 or word.find("@R_") == 0:
            ok.add(word)
        else: non_ok.add(word)
    return ok, non_ok

def process_parallel(sf, tf, of, sv, tv):
    with open(of, 'w') as fout:
        with open(sf) as sin:
            with open(tf) as tin:
                #for sline, tline in itertools.izip(sin, tin): #for Python 2x
                for sline, tline in zip(sin, tin): #for Python 3x
                    #print >>fout, '<s>', #for Python 2x
                    print("<s> ", end='', file=fout, flush=False) #for Python 3x
                    for token in sline.split():
                        if token in sv:
                            #print >>fout, token, #for Python 2x
                            print(token + " ", end='', file=fout, flush=False) #for Python 3x
                        else:
                            #print >>fout, '<unk>', #for Python 2x
                            print ("<unk> ", end='', file=fout, flush=False) #for Python 3x
                    #print >>fout, '</s>', '|||', #for Python 2x
                    print("</s>", end=' |||', file=fout, flush=False) #for Python 3x

                    #print >>fout, '<s>', #for Python 2x
                    print(" <s> ", end='', file=fout, flush=False) #for Python 3x
                    for token in tline.split():
                        if token in tv:
                            #print >>fout, token, #for Python 2x
                            print(token + " ", end='', file=fout, flush=False) #for Python 3x
                        else:
                            #print >>fout, '<unk>', #for Python 2x
                            print ("<unk> ", end='', file=fout, flush=False) #for Python 3x
                    #print >>fout, '</s>', #for Python 2x
                    print("</s>", file=fout, flush=False) #for Python 3x

def process_monolingual(sf, of, v):
    with open(of, 'w') as fout:
        with open(sf) as sin:
                for sline in sin:
                    #print >>fout, '<s>', #for Python 2x
                    print("<s> ", end='', file=fout, flush=False) #for Python 3x
                    for token in sline.split():
                        if token in v:
                            #print >>fout, token, #for Python 2x
                            print(token + " ", end='', file=fout, flush=False) #for Python 3x
                        else:
                            #print >>fout, '<unk>', #for Python 2x
                            print ("<unk> ", end='', file=fout, flush=False) #for Python 3x
                    #print >>fout, '</s>', #for Python 2x
                    print("</s>", file=fout, flush=False) #for Python 3x

#-----------------------------------------------------------------------------------------
sfname = '../data/multi30k/train.en.atok'
tfname = '../data/multi30k/train.de.atok'

source_vocab, non_source_vocab = threshold_vocab(sfname, 2) #word freq cut-off 2
target_vocab, non_target_vocab = threshold_vocab(tfname, 2) #word freq cut-off 2

print("Source vocab size: " + str(len(source_vocab)))
print("Target vocab size: " + str(len(target_vocab)))

ofname = '../data/multi30k/train.en-de.atok.capped'

process_parallel(sfname, tfname, ofname, source_vocab, target_vocab) #train
process_parallel('../data/multi30k/val.en.atok', '../data/multi30k/val.de.atok', '../data/multi30k/val.en-de.atok.capped', source_vocab, target_vocab) #dev
process_monolingual('../data/multi30k/val.en.atok', '../data/multi30k/val.en.atok.capped', source_vocab) #dev
process_monolingual('../data/multi30k/test.en.atok', '../data/multi30k/test.en.atok', source_vocab) #test

