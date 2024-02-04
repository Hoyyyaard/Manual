SCRIPT=${1:-"scripts/umass/instrp2p_finetune.sh"}
NUM_GPUS_PER_NODE=${2:-2}
NUM_NODES=${3:-1}
JOB_ID=${4:-"instrp2p"}
LOOP_COUNTER=0

while true; do
    echo "Loop counter: $LOOP_COUNTER"
    srun -J la-g --gres=gpu:$NUM_GPUS_PER_NODE --cpus-per-task=4 -N $NUM_NODES --mem=300G \
    --time 08:00:00 \
    --nodelist superpod-gpu[001-002] --nodelist umd-cscdr-gpu[001-002]  \
    -p gpu-preempt  \
    --pty bash $SCRIPT $NUM_GPUS_PER_NODE $NUM_NODES $JOB_ID 
    
    sleep 10
    LOOP_COUNTER=$((LOOP_COUNTER+1))
done
