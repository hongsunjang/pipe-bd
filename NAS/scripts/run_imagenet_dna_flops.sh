#!/bin/bash


OUTPUT_PATH="configs/imagenet_dna_flops_mullog"
echo $OUTPUT_PATH"/learned_net"

GPUS_PER_NODE=4

MASTER_ADDR=localhost
MASTER_PORT=6007
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


torchrun $DISTRIBUTED_ARGS \
	imagenet_arch_search.py  \
	--path $OUTPUT_PATH \
	--train_method "dna" \
	--n_epochs 0 \
	--target_hardware 'flops' \
	--grad_reg_loss_type 'mul#log' \
	--grad_reg_loss_alpha 1 \
	--init_lr 0.005 \
	--grad_reg_loss_beta 3.6 

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --max_restarts 3"

if [ $? -eq 0 ];then
	OUTPUT_PATH=$OUTPUT_PATH"/learned_net"
	torchrun $DISTRIBUTED_ARGS \
		imagenet_run_exp.py  \
		--train \
		--dropout 0.2 \
		--path $OUTPUT_PATH | tee $OUTPUT_PATH'log.txt'

else
	echo 'Architecture search failed'

fi

	




