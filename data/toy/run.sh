python -m sockeye.train --source train.en.tok --target train.de.tok --source-metadata train.en.deps --validation-source val.en.tok --validation-target val.de.tok --val-source-metadata val.en.deps --use-cpu --use-gcn --output toy_model --batch-size 2 --no-bucketing --rnn-num-hidden 32 --num-embed 32
rm -rf toy_model
