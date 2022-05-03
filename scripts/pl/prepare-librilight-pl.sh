layer=$2
pool_k=$3
pool_s=$4
num_clusters=$5

ngpu=$6
nshard_per_gpu=$7
nshard=$((ngpu * nshard_per_gpu))

vocab_size=30000

echo "nshard=$nshard"

kmean_portion=0.1

case $1 in
hubert-large)
ckpt_path=/persist/git/fairseq-hubert/save/pretrained/hubert_large_ll60k.pt
feature_name=hubert_large_ll60k-l${layer}-k${pool_k}s${pool_s}-fp16-ls${kmean_portion}
;;
hubert-base)
ckpt_path=/persist/git/fairseq-hubert/save/pretrained/hubert_base_ls960.pt
feature_name=hubert_base-l${layer}-k${pool_k}s${pool_s}-fp16-ls${kmean_portion}
;;
w2v2-base)
ckpt_path=/persist/git/fairseq-hubert/save/pretrained/wav2vec_small.pt
feature_name=w2v2_base-l${layer}-k${pool_k}s${pool_s}-fp16-ls${kmean_portion}
;;
w2v2-large)
ckpt_path=/persist/git/fairseq-hubert/save/pretrained/wav2vec_vox_new.pt
feature_name=w2v2_large_ll60k-l${layer}-k${pool_k}s${pool_s}-fp16-ls${kmean_portion}
;;
*)
    echo unknown $1
    exit
;;
esac

feature_dir=features/${feature_name}/librispeech/train-${kmean_portion}

label_dir=labels/${feature_name}/c${num_clusters}
cluster_model_path=${label_dir}/km.pkl

manifest_dir=manifest/librilight-60k

ll_label_dir=labels/ll-60k/${feature_name}/c${num_clusters}
mkdir -p $ll_label_dir

echo $cluster_model_path
if [ -f $cluster_model_path ]; then
    echo "using existing k-means $cluster_model_path"

    for split in valid train; do 
        echo "dumping $split labels"
        if ! [ -f $ll_label_dir/${split}.km ]; then
            for rank in $(seq 0 $((nshard - 1))); do
                gpu_id=$((rank % ngpu))
                CUDA_VISIBLE_DEVICES=$gpu_id python tools/generate_pseudo_language.py dump_hubert_clusters \
                        --ckpt_path $ckpt_path \
                        --km_path $cluster_model_path \
                        --manifest ${manifest_dir}/${split}.tsv \
                        --rank $rank --nshard $nshard --lab_dir $ll_label_dir \
                        --pool_k $pool_k --pool_s $pool_s --fp16 True &

            done
            wait

            echo "marging $split shards"
            for rank in $(seq 0 $((nshard - 1))); do
            cat $ll_label_dir/${split}_${rank}_${nshard}.km
            done > $ll_label_dir/${split}.km
        fi
    done

else
    echo "$cluster_model_path doesn't exist"
fi

for split in train valid; do
if ! [ -f ${ll_label_dir}/${split}.chrd.km ]; then
    echo "deduplicating $split"
    python tools/convert_pseudo_language.py convert ${ll_label_dir}/${split}.km ${ll_label_dir}/${split}.chrd.km
fi
done

# if ! [ -f ${label_dir}/bpe-tokenizer-vocab${vocab_size}.json ]; then
# echo "create bpe with vocab=${vocab_size}"
# python tools/convert_pseudo_language.py train-tokenizer ${label_dir}/train.chrd.km ${label_dir}/bpe-tokenizer-vocab${vocab_size}.json --vocab_size $vocab_size
# fi

for split in train valid; do
    if ! [ -f ${ll_label_dir}/${split}.chrd_bpe${vocab_size}.km ]; then
        echo "tokenizing data"
        python tools/convert_pseudo_language.py tokenize ${label_dir}/bpe-tokenizer-vocab${vocab_size}.json  ${ll_label_dir}/${split}.chrd.km ${ll_label_dir}/${split}.chrd_bpe${vocab_size}.km
    fi
done

if ! [ -f ${ll_label_dir}/dict.chrd_bpe${vocab_size}.km.txt ]; then
    echo "copy dict"
    cp ${label_dir}/dict.chrd_bpe${vocab_size}.km.txt ${ll_label_dir}/dict.chrd_bpe${vocab_size}.km.txt
fi
