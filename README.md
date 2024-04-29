# Codigo utilizado para treinar redes de detecção

Este código foi desenvolvido com o intuito de facilitar a junção de várias redes neurais para o treino de detecção de objetos.
## Estrutura das Pastas
```
├── dataset
│   └── all
│       ├── filesJSON
│       └── train
├── Results
├── src
│   └── Detectors
│       ├── FasterRCNN
│       ├── MMdetection
│       └── YOLOV8
└── Utils
```
### Dataset
Pasta onde ficarão todas as suas imagens e suas respectivas anotações.

### Results
Pasta onde estarão os resultados das redes e seus gráficos.

### src
Pasta onde estarão os códigos que irão rodar as redes.

### Detectors
Esta pasta está sendo utilizada para a organização dos códigos. Dentro de cada pasta, há códigos referentes a cada rede.

### Utils
Nesta pasta, teremos scripts para gerar gráficos, instalação das dependências, entre outros códigos de utilidade.

## Instalações

conda create --name detectores python=3.9.16 -y

conda activate detectores

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -U openmim

mim install "mmengine>=0.7.0"

mim install "mmcv>=2.0.0rc4"

cd mmdetection

pip install -e .

pip install scikit-learn

pip install funcy

pip install albumentations

pip install ultralytics

pip install supervision==0.1.0

### Caso prefira, você pode executar o script Install.sh
Que estará em Utils
```
Utils/
├── geraDobras.py
└── Install.sh
```

```
bash Install.sh
```
## Como Usar

Primeiro, você deve baixar um dataset no formato COCO e colocá-lo na pasta "dataset/all", conforme mostrado no exemplo abaixo.

```
dataset/
└── all
    └── train
```

Em seguida, execute o script geraDobras.py que está na pasta Utils.

```
Utils/
├── geraDobras.py
└── Install.sh
```

```
python geraDobras.py
```

Após isso, a sua estrutura deverá ficar assim:

```
dataset/
└── all
    ├── filesJSON
    └── train
```

Segundo voce ira escolher os seus modelos de treinamento temos os segintes modelos YOLOV8,FasterRCNN e os modelos da mmdetection.

### YOLOV8
```
src/Detectors/
└── YOLOV8
    ├── DetectionsYolov8.py
    ├── GeraLabels.py
    ├── ModelYOLOV8.py
    ├── RunYOLOV8.py
    └── TreinoYOLOV8.sh
```

Para configurar os parâmetros da YOLOv8, você irá entrar no código ModelYOLOV8.py e alterar os parâmetros de acordo com suas necessidades.

### FasterRCNN
```
src/Detectors/
└── FasterRCNN
    ├── config.py
    ├── custom_utils.py
    ├── datasets.py
    ├── GeraDobras.py
    ├── inference.py
    ├── modelFasterRCNN.py
    ├── model.py
    ├── RunFaster.py
    └── TreinoFaster.sh
```

Para configurar os parâmetros da FasterRCNN, você irá entrar no código 
config.py e alterar os parâmetros de acordo com suas necessidades. Caso queira trocar o optmizador desta rede voce deve trocar a linha optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005) do codigo modelFasterRCNN.py


### mmdetection

Para rodar a mmdetecion é um pouco mas complexo primeiro iremos escolher um modelo que eles fornecem que esta na pasta config no diretorio que ire mostar abaixo 

```
src/Detectors/
└── MMdetection
    ├── checkpoints
    └── mmdetection
        └── configs
            ├── albu_example
            ├── atss
            ├── autoassign
            ├── _base_
            │   ├── datasets
            │   └── schedules
            ├── boxinst
            ├── bytetrack
            ├── carafe
            ├── cascade_rcnn
            ├── cascade_rpn
            ├── centernet
TEM MAIS MODELOS. BASTA ABRIR A PASTA "CONFIGS"
```
Após escolher os modelos, você irá no arquivo ResdesMM.txt, dar um Ctrl + F e verificar se o seu modelo foi encontrado. Certifique-se de que os nomes são iguais. Se for o caso, não haverá problema. Caso não esteja no arquivo, tente escolher outro modelo.
```
.
└── RedesMM.txt
```
Para configurar os parâmetros da mmdetection, você irá entrar no código 
ConfigTrain.py e alterar os parâmetros de acordo com suas necessidades.
```
src/Detectors/
└── MMdetection
    ├── CheckPoint.py
    ├── ConfigTrain.py
    ├── MMdetector.py
    ├── RunMMdetecion.py
    ├── train.py
    └── Treinommdetection.sh
```
### Rodar o treino

Logo após você ter configurado e escolhido os modelos, você irá abrir o código configDetectores.py. Lá você terá uma variável chamada MODEL. Para treinar os modelos escolhidos, basta modificar essa variável. Por exemplo:

```
MODELS = ['YOLOV8','MMdetections/sabl-faster-rcnn_r50_fpn_1x_coco','MMdetections/detr_r50_8xb2-150e_coco','MMdetections/fovea_r50_fpn_4xb4-1x_coco']
```
Note que para usar os modelos da mmdetection, temos que colocar MMDetections/nome_do_modelo.

Pronto Agora é so rodar o configDetectors.py que está em src

```
src/
├── configDetectors.py
└── RestultsDetections.py
```

```
python configDetectors.py
```
