__conda_setup="$('/home/crslab/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/crslab/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/home/crslab/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/home/crslab/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate GSAM
cd ~/Grounded-Segment-Anything

python infer_pointclouds.py --num_views 5
conda deactivate

conda activate se3dif_env
cd ~/cehao/grasp_diffusion
python scripts/sample/demo_sample_test.py --inpaint
conda deactivate

# conda activate GSAM
# conda deactivate