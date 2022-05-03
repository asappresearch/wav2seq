nshard=4

kmean_portion=0.1
# ckpt_path=/persist/git/fairseq-hubert/save/pretrained/hubert_base_ls960.pt
ckpt_path=$1
layer=$2
pool_k=$3
pool_s=$4
num_clusters=$5
# pretrain_manifest=manifest/librispeech/train-960/${split}.tsv
pretrain_manifest=manifest/librispeech/train-960/${split}.tsv

feature_name=hubert_base-l${layer}-k${pool_k}s${pool_s}-fp16-ls${kmean_portion}

feature_dir=features/${feature_name}/librispeech/train-${kmean_portion}

mkdir -p $feature_dir 

split=train-${kmean_portion}
for rank in $(seq 0 $((nshard - 1))); do
    if [ -f ${feature_dir}/${split}_${rank}_${nshard}.npy ] ; then
        echo "${feature_dir}/${split}_${rank}_${nshard}.npy exists"
    else
        python tools/generate_pseudo_language.py dump_hubert_features \
                --ckpt_path $ckpt_path \
                --manifest $pretrain_manifest  --rank $rank --nshard $nshard --feat_dir $feature_dir \
                --layer $layer \
                --pool_k $pool_k --pool_s $pool_s --fp16 True &
    fi
done
wait


label_dir=labels/${feature_name}/c${num_clusters}
cluster_model_path=${label_dir}/km.pkl
mkdir -p $label_dir

echo $cluster_model_path
if [ -f $cluster_model_path ]; then
    echo "using existing k-means $cluster_model_path"
else
    echo "training k-means model with $num_clusters clusters"
    python tools/learn_kmeans.py $feature_dir train-${kmean_portion} $nshard $cluster_model_path $num_clusters --percent 1
fi

for split in valid train; do 
    echo "dumping $split labels"
    [ -s ${label_dir}/${split}.km ] || rm -f ${label_dir}/${split}.km
    if ! [ -f ${label_dir}/${split}.km ]; then
        for rank in $(seq 0 $((nshard - 1))); do
            python tools/generate_pseudo_language.py dump_hubert_clusters \
                    --ckpt_path $ckpt_path \
                    --km_path $cluster_model_path \
                    --manifest manifest/librispeech/train-960/${split}.tsv \
                    --rank $rank --nshard $nshard --lab_dir $label_dir \
                    --layer $layer \
                    --pool_k $pool_k --pool_s $pool_s --fp16 True &

        done
        wait

        echo "marging $split shards"
        for rank in $(seq 0 $((nshard - 1))); do
        cat $label_dir/${split}_${rank}_${nshard}.km
        done > $label_dir/${split}.km
    fi
done

echo "create dict"
python tools/create_dict.py ${label_dir}/train.km ${label_dir}/dict.km.txt


echo "done"
