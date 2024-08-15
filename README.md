# Codigo utilizado para treinar redes de detecção

Este código foi desenvolvido com o intuito de facilitar a junção de várias redes neurais para o treino de detecção de objetos.
## Estrutura das Pastas
```
├── dataset
│   └── all
│       ├── filesJSON
│       └── train
├── results
├── src
│   └── detectors
│       ├── FasterRCNN
│       └── YOLOV8
└── utils
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

### Copie e cole linha por linha no terminal

```
$ conda create --name detectores python=3.9.16 -y

$ conda activate detectores

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

$ pip install scikit-learn

$ pip install funcy

$ pip install albumentations

$ pip install ultralytics

$ pip install supervision==0.1.0

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
utils/
├── geraDobras.py
└── Install.sh
temos os seguintes argumentos -folds e -valperc
-folds determina a quantidade de dobras a ser criada o por padrão é 5 dobras
-valperc determina o percentual de imagens a ser usado para validação por padrão é 0.3
```

```
cd utils
python geraDobras.py
```

Após isso, a sua estrutura deverá ficar assim:

```
dataset/
└── all
    ├── filesJSON
    └── train
```

Segundo voce ira escolher os seus modelos de treinamento temos os segintes modelos YOLOV8 e FasterRCNN.

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
config.py e alterar os parâmetros de acordo com suas necessidades. Caso queira trocar o optmizador desta rede voce deve trocar a linha optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005) do codigo trainFasterRCNN.py

### Rodar o treino

Logo após você ter configurado e escolhido os modelos, você irá abrir o código configDetectores.py. Lá você terá uma variável chamada MODEL. Para treinar os modelos escolhidos, basta modificar essa variável. Por exemplo:

```
MODELS = ['YOLOV8','FasterRCNN']
```

Pronto Agora é so rodar o configDetectors.py que está em src

```
src/
├── configDetectors.py
└── RestultsDetections.py
```

```
cd src
python configDetectors.py
```
