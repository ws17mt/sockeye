# download and preprocess the data
echo "Downloading the data..."
echo "Targeting the Multi30k dataset..."
mkdir -p ../data
mkdir -p ../data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C ../data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C ../data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C ../data/multi30k && rm mmt16_task1_test.tgz

# preprocess the data
echo
echo "Tokenizing the data..."
for l in en de; do for f in ../data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in ../data/multi30k/*.$l; do perl /pylon2/ci560op/fosterg/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done

# wrap the data with <s> </s> <unk>
echo
echo "Wrapping the data..."
python dict.py

