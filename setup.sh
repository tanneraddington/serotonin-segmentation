
conda create -n env_pytorch
conda activate env_pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install cython

pip uninstall opencv-python

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install opencv-python
cd ..
python train.py