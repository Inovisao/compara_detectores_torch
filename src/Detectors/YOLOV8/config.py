import os
from ultralytics import YOLO

#https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treiono

model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 't', 'yes', 'y'}


# Função para Rodar o Treino da YOLOV8
def treino():
    # Support dynamic data.yaml path for tiled datasets
    data_yaml = os.getenv('YOLOV8_DATA_YAML', '../dataset/all/data.yaml')

    model.train(
        data=data_yaml,
        epochs=_env_int('YOLOV8_EPOCHS', 1000),  # Epocas que o Modelo ira Rodar
        imgsz=_env_int('YOLOV8_IMGSZ', 640),  # Dimeção das imagens
        patience=_env_int('YOLOV8_PATIENCE', 100),  # paciencia para o modelo parar o treinamento geral mente se usa 10% das epocas
        batch=_env_int('YOLOV8_BATCH', 64),  # Tamanho do lote da GPU
        project=os.getenv('YOLOV8_PROJECT', 'YOLOV8'),  # Nome do Projeto
        exist_ok=_env_bool('YOLOV8_EXIST_OK', True),  # Caso o arquivo ja exista ele sobre escreve
        optimizer=os.getenv('YOLOV8_OPTIMIZER', 'AdamW'),  # Optimizador do modelo (SGD, Adam, AdamW, NAdam, RAdam, RMSPro) Talvez tenha mais
        single_cls=_env_bool('YOLOV8_SINGLE_CLS', False),  # Se o dataset é multiclasses = False ou Com uma classe so = True
        rect=_env_bool('YOLOV8_RECT', False),
        cos_lr=_env_bool('YOLOV8_COS_LR', True),
        lr0=_env_float('YOLOV8_LR0', 0.0001),  # Taxa De Aprendizado Inicial
        lrf=_env_float('YOLOV8_LRF', 0.001),  # Taxa de Aprendizado Final
        plots=_env_bool('YOLOV9_PLOTS', True),  # Usado para salvar os dados do treinamento para salver = True
    )


treino()