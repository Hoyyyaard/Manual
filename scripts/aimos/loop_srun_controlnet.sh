export HOME=/gpfs/u/home/AICD/AICDzhnf/scratch/
NP=4
SCRIPT=${1:-"scripts/aimos/controlnet_finetune.sh ${NP}"}
NUM_GPUS_PER_NODE=${2:-4}
NUM_NODES=${3:-1}
JOB_ID=${4:-"contorlnet"}
LOOP_COUNTER=0

while true; do
    echo "Loop counter: $LOOP_COUNTER"
    srun -J contorlnet --gres=gpu:$NUM_GPUS_PER_NODE --cpus-per-task=64 -N $NUM_NODES --mem=500G \
    --time 06:00:00 \
    --pty bash $SCRIPT $NUM_GPUS_PER_NODE $NUM_NODES $JOB_ID 
    sleep 10

    LOOP_COUNTER=$((LOOP_COUNTER+1))
done
