pretrained_ckpt=`realpath $1`

root=wav2seq-baseline

user_dir=`pwd`/wav2seq


export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

function train {
    case $split in
    ls-10h)
        data=`pwd`/manifest/librispeech
        task=ft-10h
        config=ft_w2s_10h.yaml
        train_subset=train-10
        valid_subset=dev-other
    ;;
    ls-100h)
        data=`pwd`/manifest/librispeech
        task=ft-100h
        config=ft_w2s_100h.yaml
        train_subset=train-clean-100
        valid_subset=dev-other
    ;;
    *)
        echo "unknown task ft-$split"
        exit
    ;;
    esac

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
        dataset.max_tokens=$((800000 * $batch_scale)) \
        optimization.update_freq="[1]" \
        task.labels=$labels \
        +task.label_dir=$data \
        optimization.lr="[$lr]" \
        +model.pretrained_ckpt="$pretrained_ckpt" \
        model.mask_prob=$mask_prob \
        model.mask_length=$mask_length \
        checkpoint.save_interval_updates=$save_interval_updates \
        optimization.max_update=$max_update \
        lr_scheduler.phase_ratio="$phase_ratio" \
        +optimizer.use_old_adam=false \
        model.decoder_layerdrop=$decoder_layerdrop \
        task.normalize=$normalize \
        --config-dir config/finetuning \
        --config-name $config

    if [ $use_local = true ]; then
        kill $watch_pid
        rsync -avh ${save} $nfs_folder --delete
        pkill -9 watch
    fi

}

pre_tag=${pretrained_ckpt##*/} # get basename (/my/path/to/file.pt --> file.pt)
batch_scale=1
lr=5e-5
seed=1

save_interval_updates=5000

normalize=false
ngpu=8
labels=bpe
mask_prob=0.15
mask_length=5
decoder_layerdrop=0.05

case $2 in
ft-ls-10h)

phase_ratio="[0.1, 0.4, 0.5]"
max_update=20000

split=ls-10h

tag=${pre_tag}-ft-${labels}-${split}-lr${lr}-${ngpu}gpu-mp${mask_prob}-ml${mask_length}-mt800K-update20K_0.1_0.4-dec_ld${decoder_layerdrop}-s${seed}
train

done

;;
ft-ls-100h)

phase_ratio="[0.025, 0.475, 0.5]"
max_update=80000

split=ls-100h

tag=${pre_tag}-${ckpt}-ft-${labels}-${split}-lr${lr}-${ngpu}gpu-mp${mask_prob}-ml${mask_length}-mt800K-update80K_0.025_0.475-dec_ld${decoder_layerdrop}-s${seed}
train


;;
*)
echo "unknown $1"
esac

