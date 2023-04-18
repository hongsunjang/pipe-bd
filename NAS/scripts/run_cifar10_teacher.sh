#!/bin/bash


OUTPUT_PATH="teachers/tmp"
echo $OUTPUT_PATH"/learned_net"

GPUS_PER_NODE=4

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --max_restarts 3"

if [ $? -eq 0 ];then
	OUTPUT_PATH=$OUTPUT_PATH"/learned_net"
	torchrun $DISTRIBUTED_ARGS \
		cifar10_train_teacher.py  \
		--path $OUTPUT_PATH \
		--train \

else
	echo 'Architecture search failed'

fi

	




