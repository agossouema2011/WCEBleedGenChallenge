#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:1
#SBATCH --time=165:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10

#SBATCH --mem=12000
#SBATCH --job-name="trainYOLO_WCE_Bleeding"
#SBATCH --output=Train_YOLO_ResultPrint.out

#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR\n"
echo " the name of the job is: $SLURM_JOB_NAME\n"
echo "Th job ID is $SLURM_JOB_ID\n"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST\n"
echo "Number of nodes: $SLURM_JOB_NUM_NODES\n"
echo "We are using $SLURM_CPUS_ON_NODE cores\n"
echo "We are using $SLURM_CPUS_ON_NODE cores per node\n"
echo "Total of $SLURM_NTASKS cores\n"

source ../env/bin/activate
module purge
module load fosscuda/2018b

python yolov5/train.py --img 224 --batch 16 --epochs 400 --data dataset.yaml --weights yolov5s.pt 

uname -a