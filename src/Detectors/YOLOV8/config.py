from ultralytics import YOLO
import sys

#https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treiono

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# Função para Rodar o Treino da YOLOV8
def treino():
    try:
        model.train(
            data = '../dataset/all/data.yaml',
            epochs=30, # Epocas que o Modelo ira Rodar
            imgsz=640, # Dimeção das imagens
            patience = 5, # paciencia para o modelo parar o treinamento geral mente se usa 10% das epocas
            batch = 16, # Tamanho do lote da GPU
            project = 'YOLOV8', # Nome do Projeto
            exist_ok = True, # Caso o arquivo ja exista ele sobre escreve
            optimizer = 'SGD', # Optimizador do modelo (SGD, Adam, AdamW, NAdam, RAdam, RMSPro) Talvez tenha mais
            single_cls = True, # Se o dataset é multiclasses = False ou Com uma classe so = True
            rect = False,
            cos_lr = True,
            lr0 = 0.001, # Taxa De Aprendizado Inicial
            lrf = 1.0,# Taxa de Aprendizado Final
            plots = True, # Usado para salvar os dados do treinamento para salver = True 
        )
        print("Treinamento YOLOv8 finalizado com sucesso!")
    except FileNotFoundError as e:
        print(f"[ERRO] Arquivo não encontrado: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERRO] Falha no treinamento YOLOv8: {e}")
        sys.exit(1)

try:
    treino()
except Exception as e:
    print(f"[ERRO FATAL] Erro inesperado ao rodar o script de treino YOLOv8: {e}")
    sys.exit(1)