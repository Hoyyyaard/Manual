OUTPUT=results/instrp2p/
SCRIPT=${1:-"~/x64/anaconda3/envs/manual/bin/python src/ControlNet/tutorial_train_sd21.py"}
NUM_GPUS_PER_NODE=${2:-2}
NUM_NODES=${3:-8}
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
