# w2v_path=`realpath $1`
ngpu=8

seed=1

root=wav2seq-baseline

config=w2s_librispeech_pt
train_subset=train
valid_subset=valid

user_dir=`pwd`/wav2seq


export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

function train {
    task=pt

    save=`pwd`/save-${task}/${root}/${tag}
    tb_save=`pwd`/tb-logs-${task}/${root}/${tag}

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
        model.w2v_path="$hubert_path" \
        model.mask_prob=$mask_prob \
        task.normalize=$normalize \
        model.decoder_layerdrop=$decoder_layerdrop \
        checkpoint.save_interval_updates=2500 \
        --config-dir config/pretraining \
        --config-name $config

}

lr=2e-4


case $1 in
wav2seq-hubert-base-ls960)
data=`realpath manifest/librispeech/train-960`

decoder_layers=6
head=12
max_tokens=1400000
max_update=25000
mask_prob=0.65
decoder_layerdrop=0.05
normalize=false

labels=chrd_bpe${num_bpe}.km

hubert_path=`realpath save/pretrained/hubert_base_ls960.pt`

num_clusters=500
num_bpe=30000
feat_name=hubert_base-l9-k2s2-fp16-ls0.1
label_dir=`realpath labels/${feat_name}/c${num_clusters}`
labels=chrd_bpe${num_bpe}.km



tag=wav2seq-hubert-base-ls960
train

;;
wav2seq-hubert-large-ll60k)


data=`realpath manifest/librilight-60k`

max_update=25000
max_tokens=1200000
lr=2e-4
encoder_layers=24
decoder_layers=6
head=16

normalize=true

num_clusters=500
num_bpe=30000
decoder_layerdrop=0.05
hubert_path=`realpath save/pretrained/hubert_large_ll60k.pt`
feat_name=hubert_base-l9-k2s2-fp16-ls0.1
label_dir=`realpath labels/ll-60k/${feat_name}/c${num_clusters}`
labels=chrd_bpe${num_bpe}.km

tag=wav2seq-hubert-large-ll60k
train

;;
*)
echo "unknown $1"
esac
