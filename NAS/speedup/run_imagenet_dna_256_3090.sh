#!/bin/bash


OUTPUT_PATH="configs/imagenet_dna_time_256_3090"
echo $OUTPUT_PATH"/learned_net"

GPUS_PER_NODE=4

MASTER_ADDR=localhost
MASTER_PORT=6018
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


torchrun $DISTRIBUTED_ARGS \
	imagenet_arch_search.py  \
	--path $OUTPUT_PATH \
	--train_method "dna" \
	--init_lr 0.005 \
	--n_epochs 2 \
	--print_frequency 5 | tee $OUTPUT_PATH"/log.txt"

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --max_restarts 3"
	




