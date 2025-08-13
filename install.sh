#!/bin/bash

# Nome do ambiente Conda
ENV_NAME="detectores"

# Lista de comandos a serem executados
commands=(
    "conda create --name $ENV_NAME python=3.9.16 -y"
    "conda activate $ENV_NAME"
    "conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y"
    "pip install scikit-learn funcy albumentations==1.4.4 ultralytics==8.2.87 supervision==0.1.0 pycocotools torchinfo vision-transformers torchmetrics"
    "pip install openmim==0.3.9"
    "pip install yapf==0.40.1"
    "mim install mmengine==0.10.7"
    "mim install mmcv==1.3.17"
    "mim install mmcv-full==1.7.2"
    "mim install mmdet==2.28.2"
)

# Executa cada comando da lista
for cmd in "${commands[@]}"; do
    echo "Executando: $cmd"
    eval $cmd
    # Verifica se o comando falhou
    if [ $? -ne 0 ]; then
        echo "Erro ao executar: $cmd"
        exit 1
    fi
done

echo "Todos os comandos foram executados com sucesso!"
