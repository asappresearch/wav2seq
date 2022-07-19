ngpu=8

seed=1

root=1021/102721/s2s-hubert-ll60k

config=w2s_librispeech_pt
train_subset=train
valid_subset=valid
use_local=true

user_dir=`pwd`/pseudo_language


export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

function train {
    task=pt

    if [ $use_local = true ]; then
        echo "use local and sync to NFS"
        save=/home/ubuntu/save-${task}/${root}/${tag}
        tb_save=`pwd`/tb-logs-${task}/${root}/${tag}
        nfs_folder=save-${task}/${root}
        nfs_save=${nfs_folder}/${tag}
        mkdir -p $save $tb_save $nfs_folder

        if [ -d ${nfs_save} ]; then
            echo "copying from NFS to local"
            # rsync -avh ${nfs_save}/ $save
        fi

        cmd="watch -n 60 rsync -avh ${save} $nfs_folder"
        echo "start this cmd to sync local to nfs:"
        echo $cmd
        echo $cmd >> sych_run.sh
        eval "$cmd &>/dev/null" &
        watch_pid=$!
    else
        save=`pwd`/save-${task}/${root}/${tag}
        tb_save=`pwd`/tb-logs-${task}/${root}/${tag}
    fi

    fairseq-hydra-train \
        hydra.run.dir=$save \
        hydra.output_subdir=$save \
        common.user_dir=$user_dir \
        common.tensorboard_logdir=$tb_save \
        common.log_interval=100 \
        task.data=$data \
        dataset.train_subset=$train_subset \
        dataset.valid_subset=$valid_subset \
        distributed_training.distributed_world_size=$ngpu \
        common.seed=$seed \
        dataset.max_tokens=$max_tokens \
        optimization.update_freq="[$((64 / $ngpu))]" \
        task.labels=$labels \
        +task.label_dir=$label_dir \
        model.decoder_layers=$decoder_layers \
        model.decoder_attention_heads=$head \
        model.decoder_embed_dim=$((head * 64)) \
        model.decoder_ffn_embed_dim=$((head * 64 * 4)) \
        model.encoder_embed_dim=$((head * 64)) \
        optimization.max_update=$max_update \
        +lr_scheduler.total_num_update=400000 \
        optimization.lr="[$lr]" \
        model.w2v_args=null \
        model.w2v_path="$w2v_path" \
        model.mask_prob=$mask_prob \
        task.normalize=$normalize \
        model.decoder_layerdrop=$decoder_layerdrop \
        checkpoint.save_interval_updates=2500 \
        +task.binarized_dataset=$binarized_dataset \
        --config-dir config/pretraining \
        --config-name $config

    if [ $use_local = true ]; then
        kill $watch_pid
        rsync -avh ${save} $nfs_folder
    fi

}

# pre_tag=${w2v_path##*/} # get basename (/my/path/to/file.pt --> file.pt)
normalize=false
encoder_layers=12
decoder_layers=6
head=12
max_update=100000
batch_scale=1
lr=2e-4
mask_prob=0.65

num_clusters=500
num_bpe=30000
label_dir=`realpath labels/hubert-iter2-l9/librispeech/train-960/km-hubert-iter2-l9-p0.1-c${num_clusters}`
labels=chrd_bpe${num_bpe}.km

case $1 in
1)
data=`realpath manifest/librilight-60k`
binarized_dataset=false

max_update=100000
max_tokens=1200000
lr=2e-4
encoder_layers=24
decoder_layers=6
head=16
normalize=true
num_clusters=500
num_bpe=30000
decoder_layerdrop=0.05
w2v_path=`realpath save/pretrained/hubert_large_ll60k.pt`
label_dir=`realpath labels/ll-60k/hubert_base-l9-k2s2-fp16-ls0.1/c${num_clusters}`
labels=chrd_bpe${num_bpe}.km
tag=s2s-hubert-large-ll60k-D$((head * 64))F$((head * 64 * 4))H${head}decL${decoder_layers}-ld${decoder_layerdrop}-lr${lr}-pl_C${num_clusters}_V${num_bpe}-mt${max_tokens}-update${max_update}
train

;;
2)
data=`realpath manifest/librilight-60k`
binarized_dataset=false

max_update=100000
max_tokens=1200000
lr=2e-4
encoder_layers=24
decoder_layers=6
head=16
normalize=true
num_clusters=500
num_bpe=30000
decoder_layerdrop=0.05
w2v_path=`realpath save/pretrained/hubert_large_ll60k.pt`
label_dir=`realpath labels/ll-60k/hubert_base-l9-k2s2-fp16-ls0.1/c${num_clusters}`
labels=chrd_bpe${num_bpe}.km
tag=s2s-hubert-large-ll60k-D$((head * 64))F$((head * 64 * 4))H${head}decL${decoder_layers}-ld${decoder_layerdrop}-lr${lr}-pl_C${num_clusters}_V${num_bpe}-mt${max_tokens}-update${max_update}-run2
train
;;
*)
echo "unknown $1"
esac