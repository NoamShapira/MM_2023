
module load miniconda/22.11.1_environmentally && conda activate CLL_2024
cd repos/MM_2023/ && python infer_annotation.py

###### gpu
bsub -q gpu-short -gpu num=1:j_exclusive=yes:gmem=20000 -R "rusage[mem=20000]" "module load miniconda/22.11.1_environmentally; conda activate CLL_2024;cd ~/repos/MM_2023/; export JUPYTER_ALLOW_INSECURE_WRITES=True; jupyter notebook --no-browser --port 28889 --ip='0.0.0.0'"
###### cpu
bsub -q new-medium -R "rusage[mem=50000]" "module load miniconda/22.11.1_environmentally; conda activate CLL_2024;cd ~/repos/MM_2023/; export JUPYTER_ALLOW_INSECURE_WRITES=True; jupyter notebook --no-browser --port 28889 --ip='0.0.0.0'"
bsub -q new-medium -n 5 -R "rusage[mem=50000] span[hosts=1]" "module load miniconda/22.11.1_environmentally; conda activate CLL_2024;cd ~/repos/MM_2023/;  python infer_annotation.py"

