# Código para Treinamento de Redes de Detecção

Este repositório foi desenvolvido para facilitar a junção de múltiplas redes neurais no treinamento de modelos de detecção de objetos.
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
Os modelos disponíveis para treinamento são **YOLOV8**, **YOLOV11**, **YOLO26**, **YOLOV5-TPH**, **FasterRCNN**, **RetinaNet**, **DETR**, **SSDLite** e **ViT** (YOLOS-small).

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
Os parâmetros podem ser ajustados em `config.py`. O backbone pode ser alterado por variável de ambiente com `FASTER_BACKBONE`.

#### RetinaNet
```
src/Detectors/RetinaNet
├── config.py
├── DetectionsRetinaNet.py
├── GeraLabels.py
└── RunRetinaNet.py
```
O treinamento utiliza modelos RetinaNet do `torchvision` (>= 0.17). O `config.py` expõe variáveis (`RETINANET_BACKBONE`, `RETINANET_EPOCHS`, `RETINANET_LR`, `RETINANET_BATCH`, etc.) e o `RunRetinaNet.py` monta um `DataLoader` COCO, executa o loop de treino básico e salva o melhor modelo em `best.pth` junto com os nomes das classes.

#### ViT (YOLOS-small)
```
src/Detectors/ViT
├── config.py        — hiperparâmetros (model_name, image_size, epochs, lr, batch…)
├── GeraLabels.py     — converte anotações COCO para o DataLoader (mapeamento de classes 0-indexado)
├── RunViT.py         — fine-tuning do `hustvl/yolos-small` via `transformers`
└── DetectionsViT.py  — inferência → [x, y, w, h, class_id, score]
```
Usa `AutoModelForObjectDetection`/`AutoImageProcessor` do pacote `transformers` para fazer fine-tuning de um detector ViT-based (YOLOS) end-to-end. O `config.py` expõe variáveis (`VIT_MODEL_NAME`, `VIT_IMAGE_SIZE`, `VIT_EPOCHS`, `VIT_BATCH`, `VIT_LR`, etc.) e o `RunViT.py` salva o melhor modelo em `best.pth` junto com os nomes das classes e o mapeamento de categorias.

> **Dependências importantes**  
> - YOLOv11: `pip install "ultralytics>=8.3.0" opencv-python numpy tqdm`  
> - RetinaNet: `pip install "torchvision>=0.17" pycocotools albumentations` (ou Detectron2 se preferir)  
> - ViT (YOLOS): `pip install "transformers>=4.48" timm` (requer `torch>=2.1`)  
> - Certifique-se de que `torch>=2.1` está instalado com suporte a CUDA.

### Backbones e Pesos da Loss

As redes baseadas em `torchvision` e o SSDLite agora têm pontos configuráveis por variável de ambiente. Os valores usados aparecem em `results/training_params.json` e nas colunas `backbone` e `loss_function` dos CSVs de resultado (`results.csv`, `resultsbyclass.csv`, `results_base.csv` e `results_finetune.csv`).

#### SSDLite
- `SSDLITE_BACKBONE`: `mobilenetv2` (padrão), `resnet18`, `gelan`,
  `convnext_tiny` ou `swin_tiny`. `convnext_tiny` e `swin_tiny` usam pesos
  ImageNet do torchvision; `gelan` inicia do zero.
- `SSDLITE_LOSS_CLASSIFICATION`: peso da loss de classificação (padrão `1.0`)
- `SSDLITE_LOSS_BBOX_REGRESSION`: peso da loss de regressão das caixas (padrão `1.0`)
- `SSDLITE_BOX_LOSS`: `ciou` (padrão), `inner_mpdiou`, `wise_iou` ou `siou`
- `SSDLITE_INNER_RATIO`: razão das caixas internas do Inner-MPDIoU (padrão `0.7`)

Exemplo:
```bash
MODELS_TO_RUN="SSDLite" \
SSDLITE_BACKBONE=resnet18 \
SSDLITE_LOSS_CLASSIFICATION=1.0 \
SSDLITE_LOSS_BBOX_REGRESSION=2.0 \
python main.py
```

#### RetinaNet
- `RETINANET_BACKBONE`: `resnet50_fpn` (padrão) ou `resnet50_fpn_v2`
- `RETINANET_LOSS_CLASSIFICATION`: peso da loss de classificação (padrão `1.0`)
- `RETINANET_LOSS_BBOX_REGRESSION`: peso da loss de regressão das caixas (padrão `1.0`)
- `RETINANET_BOX_LOSS`: `ciou` (padrão), `inner_mpdiou`, `wise_iou` ou `siou`
- `RETINANET_INNER_RATIO`: razão das caixas internas do Inner-MPDIoU (padrão `0.7`)

Exemplo:
```bash
MODELS_TO_RUN="RetinaNet" \
RETINANET_BACKBONE=resnet50_fpn_v2 \
RETINANET_LOSS_CLASSIFICATION=1.5 \
RETINANET_LOSS_BBOX_REGRESSION=1.0 \
python main.py
```

#### FasterRCNN
- `FASTER_BACKBONE`: `resnet50_fpn` (padrão), `resnet50_fpn_v2` ou `mobilenet_v3_large_fpn`
- `FASTER_LOSS_CLASSIFIER`: peso da loss do classificador (padrão `1.0`)
- `FASTER_LOSS_BOX_REG`: peso da loss de regressão das caixas (padrão `1.0`)
- `FASTER_LOSS_OBJECTNESS`: peso da loss de objectness da RPN (padrão `1.0`)
- `FASTER_LOSS_RPN_BOX_REG`: peso da loss de regressão das caixas da RPN (padrão `1.0`)

Exemplo:
```bash
MODELS_TO_RUN="Faster" \
FASTER_BACKBONE=mobilenet_v3_large_fpn \
FASTER_LOSS_CLASSIFIER=1.0 \
FASTER_LOSS_BOX_REG=2.0 \
python main.py
```

> Observação: o SSDLite e o RetinaNet salvam o backbone usado dentro do `best.pth`. No FasterRCNN, mantenha `FASTER_BACKBONE` igual no treino e na inferência, pois o checkpoint atual salva apenas os pesos.

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

---

## SSDLite + MobileNetV2

```
src/Detectors/SSDLite
├── config.py               — hiperparâmetros (epochs, lr, batch, NMS…)
├── GeraLabels.py           — converte anotações COCO para o DataLoader
├── RunSSDLite.py           — backbone MobileNetV2 + cabeça SSDLite + loop de treino
├── DetectionsSSDLite.py    — inferência PyTorch → [x, y, w, h, class_id, score]
├── export_onnx.py          — exporta best.pth para ONNX
└── onnx_predict.py         — inferência via ONNX Runtime (sem PyTorch)
```

### Exportar para ONNX

```bash
cd /home/neto/development/compara_detectores_torch

python src/Detectors/SSDLite/export_onnx.py \
    --checkpoint src/model_checkpoints/fold_1/SSDLite/best.pth \
    --output ssdlite.onnx
```

Dois arquivos são gerados:
- `ssdlite.onnx` — grafo ONNX com normalização ImageNet embutida
- `ssdlite.anchors.npy` — âncoras necessárias para decodificar as detecções

### Inferência com ONNX Runtime (app Python)

Dependências mínimas — **sem PyTorch**:

```bash
pip install onnxruntime numpy opencv-python
```

Exemplo de uso:

```python
from src.Detectors.SSDLite.onnx_predict import SSDLitePredictor
import cv2

# Carrega modelo uma vez
predictor = SSDLitePredictor(
    onnx_path="ssdlite.onnx",
    score_thresh=0.5,
    nms_thresh=0.5,
)

frame = cv2.imread("imagem.jpg")          # BGR, qualquer resolução
detections = predictor.predict(frame)
# detections: [(x1, y1, x2, y2, class_id, score), ...]

for x1, y1, x2, y2, cls, score in detections:
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(frame, f"{cls} {score:.2f}", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("resultado.jpg", frame)
```

### Inferência com PyTorch (durante avaliação)

```python
from src.Detectors.SSDLite.DetectionsSSDLite import ResultSSDLite
import cv2

frame = cv2.imread("imagem.jpg")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

detections = ResultSSDLite.result(
    frame_rgb,
    model_path="src/model_checkpoints/fold_1/SSDLite/best.pth",
    threshold=0.5,
)
# detections: [[x, y, w, h, class_id, score], ...]
```

---

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
