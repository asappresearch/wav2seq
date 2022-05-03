# for num_clusters in 500; do
# echo "num_clusters=${num_clusters}"

# label_dir="labels/hubert-iter2-l9/librispeech/train-960/km-hubert-iter2-l9-p0.1-c${num_clusters}"
# label_dir="labels/hubert-iter2-l9/covost-v2-en/train-430/km-hubert-iter2-l9-p0.1-c${num_clusters}"
# label_dir="labels/hubert_large_ll60k-l18-k1s1-fp16-ls0.1/c${num_clusters}"
label_dir=$1
vocab_size=$2
echo $label_dir

for split in train valid; do
if ! [ -f ${label_dir}/${split}.chrd.km ]; then
echo "converting $split"
python tools/convert_pseudo_language.py convert ${label_dir}/${split}.km ${label_dir}/${split}.chrd.km
fi
done

# for vocab_size in 1000 30000 50000; do
# for vocab_size in 10000; do

if ! [ -f ${label_dir}/bpe-tokenizer-vocab${vocab_size}.json ]; then
echo "create bpe with vocab=${vocab_size}"
python tools/convert_pseudo_language.py train-tokenizer ${label_dir}/train.chrd.km ${label_dir}/bpe-tokenizer-vocab${vocab_size}.json --vocab_size $vocab_size
fi

for split in train valid; do
# for split in train dev; do
    if ! [ -f ${label_dir}/${split}.chrd_bpe${vocab_size}.km ]; then
    echo "tokenizing data"
    python tools/convert_pseudo_language.py tokenize ${label_dir}/bpe-tokenizer-vocab${vocab_size}.json  ${label_dir}/${split}.chrd.km ${label_dir}/${split}.chrd_bpe${vocab_size}.km
    fi
done

echo "creating dict"
if ! [ -f ${label_dir}/dict.chrd_bpe${vocab_size}.km.txt ]; then
python tools/create_dict.py ${label_dir}/train.chrd_bpe${vocab_size}.km ${label_dir}/dict.chrd_bpe${vocab_size}.km.txt
fi

# done

# done