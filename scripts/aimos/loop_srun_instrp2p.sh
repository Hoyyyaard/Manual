export HOME=/gpfs/u/home/AICD/AICDzhnf/scratch/
OUTPUT=results/instrp2p/
SCRIPT=${1:-"scripts/aimos/instrp2p_finetune.sh ${OUTPUT}"}
NUM_GPUS_PER_NODE=${2:-1}
NUM_NODES=${3:-8}
JOB_ID=${4:-"instrp2p"}
LOOP_COUNTER=0
GEN_SCRIPT="scripts/aimos/instrp2p_generate.sh ${OUTPUT}"

while true; do
    echo "Loop counter: $LOOP_COUNTER"
    srun -J instrp2p --gres=gpu:$NUM_GPUS_PER_NODE --cpus-per-task=64 -N $NUM_NODES  --mem=500G \
    --time 06:00:00 \
    --pty bash $SCRIPT $NUM_GPUS_PER_NODE $NUM_NODES $JOB_ID 
    sleep 10
    srun -J instrp2p_gen --gres=gpu:1 --cpus-per-task=64 -N 1 --mem=300G \
    --time 00:30:00 \
    --pty bash $GEN_SCRIPT 1 1 instrp2p_gen
    sleep 10
    LOOP_COUNTER=$((LOOP_COUNTER+1))
done
