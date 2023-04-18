#!/bin/bash


OUTPUT_PATH="teachers/cifar10"
echo $OUTPUT_PATH"/learned_net"

GPUS_PER_NODE=4

MASTER_ADDR=localhost
MASTER_PORT=6003
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


torchrun $DISTRIBUTED_ARGS \
	cifar10_train_teacher.py  \
	--path $OUTPUT_PATH \
	--resume
	




