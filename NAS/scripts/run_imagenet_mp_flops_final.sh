#!/bin/bash


OUTPUT_PATH="configs/please"
echo $OUTPUT_PATH"/learned_net"

GPUS_PER_NODE=4

MASTER_ADDR=localhost
MASTER_PORT=6018
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#torchrun $DISTRIBUTED_ARGS \
#	imagenet_arch_search.py  \
#	--path $OUTPUT_PATH \
#	--train_method "mp" \
#	--init_lr 0.005 \
#	--target_hardware 'flops' \
#	--grad_reg_loss_alpha 0.2 \
#	--grad_reg_loss_beta 1.8 \
#	--n_epochs 20 | tee $OUTPUT_PATH"/log.txt"
#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --max_restarts 3"

if [ $? -eq 0 ];then
	OUTPUT_PATH=$OUTPUT_PATH"/learned_net"
	torchrun $DISTRIBUTED_ARGS \
		imagenet_run_exp.py  \
		--path $OUTPUT_PATH \
		--dropout 0.1 \
		--n_epochs 600 \
		--resume \
		--train | tee $OUTPUT_PATH"/log.txt"

else
	echo 'Architecture search failed'

fi

	




