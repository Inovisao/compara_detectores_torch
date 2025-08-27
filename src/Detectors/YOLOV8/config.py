from ultralytics import YOLO

#https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treiono

model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
# Função para Rodar o Treino da YOLOV8
def treino():

    model.train(
                data = '../dataset/all/data.yaml',
                epochs=500, # Epocas que o Modelo ira Rodar
                imgsz=640, # Dimeção das imagens
                patience = 50, # paciencia para o modelo parar o treinamento geral mente se usa 10% das epocas
                batch = 64, # Tamanho do lote da GPU
                project = 'YOLOV8', # Nome do Projeto
                exist_ok = True, # Caso o arquivo ja exista ele sobre escreve
                optimizer = 'AdamW', # Optimizador do modelo (SGD, Adam, AdamW, NAdam, RAdam, RMSPro) Talvez tenha mais
                single_cls = False, # Se o dataset é multiclasses = False ou Com uma classe so = True
                rect = False,
                cos_lr = True,
                lr0 = 0.0001, # Taxa De Aprendizado Inicial
                lrf = 0.01,# Taxa de Aprendizado Final
                plots = True, # Usado para salvar os dados do treinamento para salver = True 
    )
treino()