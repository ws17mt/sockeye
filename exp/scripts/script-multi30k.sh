# download the data
echo "Downloading the data..."
echo "Targeting the Multi30k dataset..."
mkdir -p ../data
mkdir -p ../data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C ../data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C ../data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C ../data/multi30k && rm mmt16_task1_test.tgz

# preprocess the data
echo ""
echo "Tokenizing the data..."
for l in en de; do for f in ../data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in ../data/multi30k/*.$l; do perl /pylon2/ci560op/fosterg/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done

# wrap the data with <s> </s> <unk> (optional)
echo ""
echo "Wrapping the data..."
python dict.py

# train a NMT system

# with Sockeye
#interact --ntasks-per-node=4 -t 05:00:00 #for CPU
#interact -p GPU-shared --gres=gpu:p100:2 -t 05:00:00 #for GPU
echo ""
echo "Training the NMT system..."
mkdir -p ../models
mkdir -p ../models/multi30k

mkdir -p ../models/multi30k/sockeye
rm -rf ../models/multi30k/sockeye #remove previous training files

# GPU
python -m sockeye.train  --source ../data/multi30k/train.en.atok \
                       --target ../data/multi30k/train.de.atok \
                       --validation-source ../data/multi30k/val.en.atok \
                       --validation-target ../data/multi30k/val.de.atok \
                       --word-min-count 2 \
                       --rnn-num-layers 1 \
                       --rnn-cell-type gru \
		       --rnn-num-hidden 128 \
		       --num-embed-source 128 \
		       --num-embed-target 128 \
		       --attention-type mlp \
		       --attention-num-hidden 128 \
		       --batch-size 64 \
		       --normalize-loss \
		       --dropout 0.1 \
		       --initial-learning-rate 0.001 \
		       --device-ids -1 \
		       --output ../models/multi30k/sockeye #&>/dev/null &

# CPU
python -m sockeye.train  --source exp/data/multi30k/train.en.atok --target exp/data/multi30k/train.de.atok --validation-source exp/data/multi30k/val.en.atok --validation-target exp/data/multi30k/val.de.atok --word-min-count 2 --rnn-num-layers 1 --rnn-cell-type gru --rnn-num-hidden 128 --num-embed-source 128 --num-embed-target 128 --attention-type mlp --attention-num-hidden 128 --batch-size 64 --normalize-loss --dropout 0.1 --initial-learning-rate 0.001 --use-cpu --output exp/models/multi30k/sockeye

export PYTHONPATH=$PYTHONPATH:/pylon2/ci560op/vhoang/tools/sockeye-forked
rm -rf exp/models/multi30k/sockeye-gpu
python sockeye/train.py --source exp/data/multi30k/train.en.atok --target exp/data/multi30k/train.de.atok --validation-source exp/data/multi30k/val.en.atok --validation-target exp/data/multi30k/val.de.atok --word-min-count 2 --rnn-num-layers 1 --rnn-cell-type gru --rnn-num-hidden 128 --num-embed-source 128 --num-embed-target 128 --attention-type mlp --attention-num-hidden 128 --batch-size 64 --normalize-loss --dropout 0.1 --initial-learning-rate 0.001 --device-ids -2 --output exp/models/multi30k/sockeye-gpu

# train a NMT system
# with Mantidae
#interact --ntasks-per-node=4 -t 05:00:00 #for CPU
#interact -p GPU-shared --gres=gpu:p100:2 -t 05:00:00 #for GPU

# GPU
mkdir /pylon2/ci560op/vhoang/tools/sockeye/exp/models/multi30k/mantidae
nice /pylon2/ci560op/vhoang/tools/Mantidae/build_gpu/src/attentional --dynet_mem 1000 --minibatch_size 64 --treport 100 --dreport 10000 -t /pylon2/ci560op/vhoang/tools/sockeye/exp/data/multi30k/train.en-de.atok.capped -d /pylon2/ci560op/vhoang/tools/sockeye/exp/data/multi30k/val.en-de.atok.capped -p /pylon2/ci560op/vhoang/tools/sockeye/exp/models/multi30k/mantidae/params.en-de.AM.sl_1_tl_1_h_128_a_128_gru_bidir --bidirectional --gru --slayers 1 --tlayers 1 -h 128 -a 128 -e 15 --lr_epochs 4 --lr_eta 0.1 --lr_eta_decay 2 &>/pylon2/ci560op/vhoang/tools/sockeye/exp/models/multi30k/mantidae/log.en-de.AM.sl_1_tl_1_h_128_a_128_gru_bidir &

# translate/decode a given test data

# with Sockeye

# GPU
python -m sockeye.translate --models exp/models/multi30k/sockeye-gpu --beam-size 5 --device-ids -1 < exp/data/multi30k/test.en.atok > exp/models/multi30k/sockeye-gpu/test.de.atok.translated-beam5

# CPU
python -m sockeye.translate --models exp/models/multi30k/sockeye --beam-size 5 --device-ids -1 < exp/data/multi30k/test.en.atok > exp/models/multi30k/sockeye/test.de.atok.translated-beam5

# with Mantidae
nice /pylon2/ci560op/vhoang/tools/Mantidae/build_gpu/src/attentional --dynet_mem 100,1,100 -t /pylon2/ci560op/vhoang/tools/sockeye/exp/data/multi30k/train.en-de.atok.capped -T /pylon2/ci560op/vhoang/tools/sockeye/exp/data/multi30k/test.en.atok.capped -i /pylon2/ci560op/vhoang/tools/sockeye/exp/models/multi30k/mantidae/params.en-de.AM.sl_1_tl_1_h_128_a_128_gru_bidir --bidirectional --gru --slayers 1 --tlayers 1 -h 128 -a 128 --beam 5 | sed 's/<s> //g' | sed 's/ <\/s>//g' > /pylon2/ci560op/vhoang/tools/sockeye/exp/models/multi30k/mantidae/test.de.atok.translated-beam5

# evaluate

# with sockeye
perl /pylon2/ci560op/fosterg/tools/mosesdecoder/scripts/generic/multi-bleu.perl exp/data/multi30k/test.de.atok < exp/models/multi30k/sockeye-gpu/test.de.atok.translated-beam5 > exp/models/multi30k/sockeye-gpu/test.de.atok.translated-beam5.BLEU

# with Mantidae
perl /pylon2/ci560op/fosterg/tools/mosesdecoder/scripts/generic/multi-bleu.perl exp/data/multi30k/test.de.atok < exp/models/multi30k/mantidae/test.de.atok.translated-beam5 > exp/models/multi30k/mantidae/test.de.atok.translated-beam5.BLEU

# dual learning (in sockeye)
PYTHONPATH=$SOCKEYE python sockeye/dual_learning.py --source exp/data/multi30k/train.de.atok --target exp/data/multi30k/train.en.atok --validation-source exp/data/multi30k/val.de.atok --validation-target exp/data/multi30k/val.en.atok --mono-source exp/data/multi30k/train.de.atok --mono-target exp/data/multi30k/train.en.atok --models exp/models/multi30k/de-en/ exp/models/multi30k/en-de/ exp/models/multi30k/de-de/ exp/models/multi30k/en-en/ --output exp/models/multi30k/de-en-de --output-s2t exp/models/multi30k/de-en-dl/ --output-t2s exp/models/multi30k/en-de-dl/ --max-input-len 100 --beam-size 5 --k-best 2 --initial-lr-gamma-s2t 0.0002 --initial-lr-gamma-t2s 0.02 --alpha 0.005 --epoch 15 --dev-round 10000 --device-ids -1 --overwrite-output &>/dev/null &

PYTHONPATH=$SOCKEYE python sockeye/dual_learning.py --source exp/data/multi30k/train.de.atok --target exp/data/multi30k/train.en.atok --validation-source exp/data/multi30k/val.de.atok --validation-target exp/data/multi30k/val.en.atok --mono-source exp/data/multi30k/train.de.atok --mono-target exp/data/multi30k/train.en.atok --models exp/models/multi30k/de-en/ exp/models/multi30k/en-de/ exp/models/multi30k/de-de/ exp/models/multi30k/en-en/ --output exp/models/multi30k/de-en-de --output-s2t exp/models/multi30k/de-en-dl/ --output-t2s exp/models/multi30k/en-de-dl/ --max-input-len 100 --beam-size 5 --k-best 2 --initial-lr-gamma-s2t 0.0002 --initial-lr-gamma-t2s 0.02 --alpha 0.005 --epoch 15 --dev-round 10000 --use-cpu --overwrite-output &>/dev/null &

