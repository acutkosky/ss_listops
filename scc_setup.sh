

module load python3
module load gcc
module load pytorch
module load tensorflow


python3 -m venv lra_env --system-site-packages
source lra_env/bin/activate

pip3 install -r requirements.txt



export DATA_DIR=/projectnb/aclab/cutkosky/statespaces/dss/data/lra_release/lra_release/listops/




