conda create -n neurogym_env python=3.10
conda activate neurogym_env
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
pip install torch torchaudio torchvision prettytable