from pathlib import Path

from ultralytics import YOLO

#https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treiono

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_YAML = REPO_ROOT / 'dataset' / 'all' / 'data.yaml'

model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)


def build_train_kwargs():
    return dict(
        data=str(DATA_YAML),
        epochs=1000,  # Epocas que o Modelo ira Rodar
        imgsz=640,  # Dimeção das imagens
        patience=100,  # paciencia para o modelo parar o treinamento geral mente se usa 10% das epocas
        batch=64,  # Tamanho do lote da GPU
        project='YOLOV8',  # Nome do Projeto
        exist_ok=True,  # Caso o arquivo ja exista ele sobre escreve
        optimizer='AdamW',  # Optimizador do modelo (SGD, Adam, AdamW, NAdam, RAdam, RMSPro) Talvez tenha mais
        single_cls=False,  # Se o dataset é multiclasses = False ou Com uma classe so = True
        rect=False,
        cos_lr=True,
        lr0=0.0001,  # Taxa De Aprendizado Inicial
        lrf=0.01,  # Taxa de Aprendizado Final
        plots=True,  # Usado para salvar os dados do treinamento para salver = True
        workers=0,  # Desativa workers paralelos para evitar falhas de multiprocessing
    )


# Função para Rodar o Treino da YOLOV8
def treino():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f'Arquivo de configuração do dataset não encontrado: {DATA_YAML}')
    model.train(**build_train_kwargs())


if __name__ == '__main__':
    treino()