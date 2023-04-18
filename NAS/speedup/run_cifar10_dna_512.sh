#!/bin/bash


OUTPUT_PATH="configs/cifar10_dna_time_512"
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
	--train_method "dna" \
	--dataset 'cifar10' \
	--stride_stages '1,2,2,1,2,1' \
	--n_epochs 7 \
	--n_worker 8 \
	--train_batch_size 512 \
	--path $OUTPUT_PATH  | tee $OUTPUT_PATH"/log.txt"
	#--target_hardware 'flops' \



