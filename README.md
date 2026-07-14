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
│   ├── all                   ← dataset original (anotações COCO completas)
│   │   └── _annotations.coco.json
│   ├── sahi                  ← tiles 640×640 gerados com SAHI
│   │   ├── fold_1/ … fold_5/ ← imagens train/val/test por fold
│   │   ├── fold_N_stats.json ← métricas e lista de imagens por fold
│   │   ├── filesJSON/        ← JSONs COCO por fold/split (gerado pelos scripts)
│   │   ├── filesJSON_infos/  ← YAML/stats dos folds
│   │   ├── fold_1/           ← train|val|test/images e labels
│   │   └── ...
│   ├── asahi                 ← idem, tiles com aspect-ratio fixo
│   └── asahi_rect            ← idem, tiles retangulares
├── results
├── scripts
│   ├── gen_fold_jsons.py     ← gera filesJSON COCO convertendo as labels YOLO (.txt) de cada split
│   ├── adapt_yolo_folds.py   ← compatibilidade/manifesto para datasets por fold
│   └── adapt_all_datasets.sh ← roda adapt_yolo_folds nos 3 datasets de uma vez
├── src
│   └── Detectors
│      ├── Detr
│      ├── FasterRCNN
│      ├── YOLOV8
│      ├── YOLOV11
│      └── RetinaNet
├── tests
└── utils
```

### Diretórios
- **dataset/all/**: Anotações COCO originais (`_annotations.coco.json`) usadas como fonte para geração dos filesJSON.
- **dataset/sahi|asahi|asahi_rect/**: Datasets tileados com validação cruzada de 5 dobras. O contrato principal usa `filesJSON/` na raiz e imagens por fold em `fold_N/{train,val,test}/images`. O layout achatado `train/`, `val/`, `test/` é apenas fallback legado.
- **results/**: Armazena os resultados das redes e seus gráficos.
- **scripts/**: Scripts de preparação de dataset e utilitários de pipeline.
- **src/**: Contém os códigos das redes.
- **tests/**: Testes de integridade do dataset e smoke tests de pipeline.
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

#### 1a. Dataset padrão (sem tiling)
Coloque o dataset no formato COCO em `dataset/all/` e divida em dobras:
```sh
cd utils
python geraDobras.py --folds 5 --valperc 0.3
```
Resultado:
```
dataset/all/
├── filesJSON/   ← fold_N_{train,val,test}.json
└── train/       ← imagens
```

#### 1b. Datasets tileados (SAHI)
Se os datasets já possuem as dobras geradas em `dataset/sahi/`, `dataset/asahi/` e `dataset/asahi_rect/` (com `fold_N_stats.json` e imagens por fold), execute:
```sh
# Gera os filesJSON COCO por fold/split para os 3 datasets
python scripts/gen_fold_jsons.py

# Valida o contrato antes de treinar
python scripts/validate_dataset_contract.py --root dataset/asahi_rect
```

Para rodar um dataset tileado, defina `DATASET_ROOT` antes de chamar `main.py`:
```sh
DATASET_ROOT=dataset/sahi      python src/main.py
DATASET_ROOT=dataset/asahi     python src/main.py
DATASET_ROOT=dataset/asahi_rect python src/main.py
```

Se `DATASET_ROOT` não for definido, `main.py` usa `dataset/all/` por padrão.

O contrato esperado para novos datasets tileados está em `DATASET_CONTRACT.md`.
Para `asahi_rect`, o pipeline resolve `fold_N_train.json` em
`fold_N/train/images`, `fold_N_val.json` em `fold_N/val/images` e
`fold_N_test.json` em `fold_N/test/images`. O modo recomendado é
`tiling.evaluation_mode=basic`.

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
