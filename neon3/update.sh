
# need to activate anaconda python first!!!
# ac3init
#export PATH="/home/watkinspv/anaconda2/bin:$PATH"
source deactivate
conda remove --name neon --all
conda info --envs
cd ~/gits/neon
git pull
conda create --name neon pip matplotlib
source activate neon
make sysinstall -e VIS=true
cd ../emdrp/neon3/
pip install -r requirements.txt

