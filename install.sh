
 conda create --name detectores python=3.9.16 -y
 conda activate detectores
 conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
 pip install scikit-learn
 pip install funcy
 pip install albumentations
 pip install ultralytics
 pip install supervision==0.1.0
 pip install pycocotools
 pip install torchinfo
 pip install vision-transformers

