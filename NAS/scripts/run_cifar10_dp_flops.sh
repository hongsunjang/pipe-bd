#!/bin/bash


OUTPUT_PATH="configs/cifar10_dp_flops"
echo $OUTPUT_PATH"/learned_net"

GPUS_PER_NODE=4

MASTER_ADDR=localhost
MASTER_PORT=6004
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#TRAIN_ARGS="--gpu 1,2,3,4"


#
torchrun $DISTRIBUTED_ARGS \
	imagenet_arch_search.py  \
	--path $OUTPUT_PATH \
	--train_method "dp" \
	--dataset 'cifar10' \
	--stride_stages '1,2,2,1,2,1' \
	--width_stages '24,32,64,96,160,320' \
	--weight_decay 5e-4 \
	--init_lr 0.025 \
	--n_epochs 360 \
	--target_hardware 'flops' \
	--warmup_epochs 120 \
	--grad_reg_loss_alpha 1 \
	--grad_reg_loss_beta 3.6  | tee $OUTPUT_PATH'log.txt'


#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --max_restarts 3"

if [ $? -eq 0 ];then
	OUTPUT_PATH=$OUTPUT_PATH"/learned_net"
	torchrun $DISTRIBUTED_ARGS \
		cifar10_run_exp.py  \
		--path $OUTPUT_PATH \
		--train | tee $OUTPUT_PATH'log.txt'

else
	echo 'Architecture search failed'

fi

	




