#!/bin/bash


OUTPUT_PATH="configs/cifar10_mp_ours_4dp"
echo $OUTPUT_PATH"/learned_net"

GPUS_PER_NODE=4

MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#TRAIN_ARGS="--gpu 1,2,3,4"


torchrun $DISTRIBUTED_ARGS \
	imagenet_arch_search.py  \
	--train_method "mp" \
	--dataset 'cifar10' \
	--stride_stages '1,2,2,1,2,1' \
	--width_stages '24,32,64,96,160,320' \
	--weight_decay 5e-4 \
	--init_lr 0.025 \
	--n_epochs 7 \
	--scheme 'ahd' \
	--n_worker 4 \
	--path $OUTPUT_PATH | tee $OUTPUT_PATH"/log.txt"
	#--target_hardware 'flops' \


