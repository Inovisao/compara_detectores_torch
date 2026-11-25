# Código para Treinamento de Redes de Detecção

Este repositório foi desenvolvido para facilitar a junção de múltiplas redes neurais no treinamento de modelos de detecção de objetos.

## Fine-tuning do first para o second

No ambiente Python com as dependências de treinamento instaladas, execute:

```bash
cd src
DATASET_NAME=second \
RUN_TAG=finetune_second \
MODELS_TO_RUN=YOLO26 \
YOLO26_WEIGHTS="$PWD/model_checkpoints/nano/fold_2/YOLO26/train/weights/best.pt" \
YOLO26_EPOCHS=100 YOLO26_PATIENCE=20 YOLO26_BATCH=16 \
YOLO26_LR0=0.0001 YOLO26_DEVICE=0 \
python3 main.py
```

Cada dobra do `second` começa com o mesmo checkpoint do `first` indicado acima.
Os checkpoints são salvos em `src/model_checkpoints/finetune_second/`.
Para um novo experimento, use outro `RUN_TAG`: reutilizar a etiqueta substitui seus checkpoints.

Os resultados são separados automaticamente pelo dataset:

- `results/first/results.csv` e `results/second/results.csv`;
- `counting.csv`, `resultsbyclass.csv` (quando habilitado) e `prediction/` dentro da pasta de cada dataset;
- `training_params.json` na pasta do dataset, salvo antes do treino e atualizado após cada dobra;
- `runs/<data-hora>/training_params.json` preserva os parâmetros de cada execução;
- cada checkpoint também recebe seu `training_params.json`, incluindo `training_args` do `args.yaml` efetivamente usado.

`DATASET_ROOT`, quando definido, seleciona o dataset pelo caminho e usa o nome dessa pasta para separar os resultados.
O CSV acumula execuções quando `RUN_TAG` está preenchido. Os arquivos antigos diretamente em `results/` não são importados automaticamente.
Para treinar no `first` a partir dos pesos base, use `DATASET_NAME=first`, `RUN_TAG=nano_first` e `YOLO26_WEIGHTS=yolo26n.pt` (requer o dataset `first` disponível).

Para gerar gráficos, execute de `utils/`: `Rscript graficos.R first` ou `Rscript graficos.R second`.

## Link dos codigos utilizados como base
### YOLOV8
- **https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings**:
### FasterRCNN
- **https://github.com/AarohiSingla/Faster-R-CNN-on-custom-dataset-Using-Pytorch**
### Detr
- **https://debuggercafe.com/train-detr-on-custom-dataset/**

## Estrutura de Pastas
```
├── dataset
│   └── all
│       ├── filesJSON
│       └── train
├── results
├── src
│   └── Detectors
│      ├── Detr
│      ├── FasterRCNN
│      ├── YOLOV8
│      ├── YOLOV11
│      └── RetinaNet
└── utils

```
### Diretórios
- **dataset/**: Contém as imagens e anotações no formato COCO. As imagens devem ter resolução de 640x640 e estar na pasta `train`, junto ao arquivo `coco.json`.
- **results/**: Armazena os resultados das redes e seus gráficos.
- **src/**: Contém os códigos das redes.
- **Detectors/**: Diretório para organização dos modelos de detecção.
- **utils/**: Scripts auxiliares para geração de gráficos, instalação de dependências e outras utilidades.
## Instalação

Execute os seguintes comandos no terminal para configurar o ambiente:

```sh
conda create --name detectores python=3.9.16 -y
conda activate detectores
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install scikit-learn funcy albumentations==1.4.4 ultralytics==8.2.87 supervision==0.1.0 pycocotools torchinfo vision-transformers torchmetrics tensorboard

-> Para a YOLOv11 rodar pip install ultralytics==8.3.217

obs : todas as bibliotecas utilzadas estão no arquivo Bibliotecas.yml
```

## Como Usar

### 1. Preparando o Dataset
Baixe um dataset no formato COCO e coloque-o na pasta `dataset/all` conforme a estrutura abaixo:
```
dataset/
└── all
    └── train
```

Em seguida, execute o script `geraDobras.py` na pasta `utils/` para dividir os dados em dobras:
```sh
cd utils
python geraDobras.py --folds 5 --valperc 0.3
```
*Parâmetros:*  
- `--folds`: Define a quantidade de dobras (padrão: 5).
- `--valperc`: Percentual de imagens para validação (padrão: 0.3).

Após a execução, a estrutura será:
```
dataset/
└── all
    ├── filesJSON
    └── train
```

### 2. Escolhendo e Configurando os Modelos
Os modelos disponíveis para treinamento são **YOLOV8**, **YOLOV11**, **YOLO26**, **YOLOV5-TPH**, **FasterRCNN**, **RetinaNet** e **DETR**.

#### YOLOV8
```
src/Detectors/
└── YOLOV8
    ├── DetectionsYolov8.py
    ├── GeraLabels.py
    ├── config.py
    ├── RunYOLOV8.py
    └── TreinoYOLOV8.sh
```
Altere os parâmetros do modelo no arquivo `config.py`.

#### YOLOV11
```
src/Detectors/
└── YOLOV11
    ├── DetectionsYOLOV11.py
    ├── GeraLabels.py
    ├── RunYOLOV11.py
    └── config.py
```
Utiliza o pacote `ultralytics` (>= 8.3.0) com os pesos `yolo11*.pt`. Por padrão o código baixa `yolo11s.pt`; se quiser usar um checkpoint local (ex.: `src/yolo11s.pt`), defina `YOLOV11_WEIGHTS` antes de rodar. Hiperparâmetros podem ser sobrepostos via variáveis como `YOLOV11_EPOCHS`, `YOLOV11_BATCH`, `YOLOV11_LR0`, etc. O script `RunYOLOV11.py` cria o `data_yolov11.yaml`, executa o treino e move o diretório `YOLOV11/train/` para o checkpoint da dobra.

#### YOLO26
```
src/Detectors/
└── YOLO26
    ├── DetectionsYOLO26.py
    ├── GeraLabels.py
    ├── RunYOLO26.py
    └── config.py
```
Utiliza o pacote `ultralytics` com pesos `yolo26*.pt`. Por padrão o código usa `yolo26n.pt`; você pode sobrescrever com variáveis como `YOLO26_WEIGHTS`, `YOLO26_EPOCHS`, `YOLO26_BATCH`, `YOLO26_LR0` e `YOLO26_DEVICE`. O script `RunYOLO26.py` cria o `data_yolo26.yaml`, executa o treino e move o diretório `YOLO26/train/` para o checkpoint da dobra.

#### FasterRCNN
```
src/Detectors/FasterRCNN
├── config.py
├── geradataset.py
├── inference.py
├── runFaster.py
├── train.py
└── TreinoFaster.sh
```
Os parâmetros podem ser ajustados em `config.py`. Para alterar o otimizador, modifique a linha no arquivo `train.py`:
```python
optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
```

#### RetinaNet
```
src/Detectors/RetinaNet
├── config.py
├── DetectionsRetinaNet.py
├── GeraLabels.py
└── RunRetinaNet.py
```
O treinamento utiliza o `retinanet_resnet50_fpn` do `torchvision` (>= 0.17). O `config.py` expõe variáveis (`RETINANET_EPOCHS`, `RETINANET_LR`, `RETINANET_BATCH`, etc.) e o `RunRetinaNet.py` monta um `DataLoader` COCO, executa o loop de treino básico e salva o melhor modelo em `best.pth` junto com os nomes das classes.

> **Dependências importantes**  
> - YOLOv11: `pip install "ultralytics>=8.3.0" opencv-python numpy tqdm`  
> - RetinaNet: `pip install "torchvision>=0.17" pycocotools albumentations` (ou Detectron2 se preferir)  
> - Certifique-se de que `torch>=2.1` está instalado com suporte a CUDA.

### 3. Executando o Treinamento
No arquivo `main.py`, edite a variável `MODELS` ou defina a variável de ambiente `MODELS_TO_RUN` para selecionar os modelos desejados, por exemplo:
```bash
MODELS_TO_RUN="YOLO26,YOLOV11,RetinaNet" python main.py
```

Agora, execute o treinamento manualmente (caso não use a variável de ambiente):
```sh
cd src
python main.py
```

Os resultados serão salvos na pasta `results/`.

## Adição de Novos Modelos

### 1. Estrutura de Pastas

```plaintext
src/Detectors
├── Detr
├── FasterRCNN
└── YOLOV8
```

### 2. Estrutura das Redes

Cada pasta em `src/Detectors/<Modelo>` possui, no mínimo, os arquivos abaixo:

- **config.py** – define hiperparâmetros padrão e lê sobrescritas via variáveis de ambiente.  
- **GeraLabels.py** – converte as anotações COCO para o formato consumido pelo modelo (YOLO txt, COCO padronizado / DataLoader, etc).  
- **Run<Model>.py** – prepara o dataset, chama o pipeline de treino e move os artefatos para `src/model_checkpoints/fold_<n>/<Modelo>/`.  
- **Detections<Model>.py** – executa a inferência no formato `[x, y, w, h, class_id, score]`, com `class_id` iniciando em 1.

Modelos como YOLOV8/YOLOV11 também possuem scripts auxiliares (`TreinoYOLOV*.sh`) para compatibilidade com execuções antigas.

### 3. Verificação de Dependências

Antes de rodar um novo modelo, é essencial verificar se todas as dependências necessárias estão instaladas e compatíveis com os modelos já existentes. Certifique-se de que bibliotecas como `torch`, `numpy`, `opencv`, entre outras, estejam na versão correta para evitar conflitos entre os modelos.
