import os
import subprocess
import sys
import shutil

from Detectors.FasterRCNN.geradataset import geredata


def runFaster(fold, fold_dir, ROOT_DATA_DIR):
    geredata(fold)  # Função para criar as labels do treino do Faster R-CNN

    # Remove se houver resultados antigos na pasta model_checkpoints
    if os.path.exists(os.path.join(fold_dir, 'Faster')):
        shutil.rmtree(os.path.join(fold_dir, 'Faster'))

    # A versão original executava o shell script TreinoFaster.sh, que no
    # Windows falha com "WinError 193: não é um aplicativo Win32 válido"
    # (mesmo problema já corrigido em RunYOLOV8.py). O .sh continha apenas
    # uma linha ('python Detectors/FasterRCNN/train.py'), então chamamos
    # train.py diretamente — funciona em Windows, Linux e macOS.
    #
    # IMPORTANTE: o alvo é train.py, NAO config.py. O config.py so define
    # constantes (BATCH_SIZE, LR, etc.) e termina em segundos sem treinar
    # nada -- quem de fato dispara o treino e o train.py, que faz
    # 'from config import (...)'. Esse import relativo so funciona porque,
    # ao rodar train.py DIRETAMENTE como script, o Python acrescenta a
    # pasta do proprio script (Detectors/FasterRCNN/) ao inicio do caminho
    # de busca -- por isso mantemos a chamada apontando para o arquivo .py
    # em vez de importa-lo como modulo.
    #
    # sys.executable garante o MESMO interpretador que está rodando o
    # main.py (o .sh chamava 'python', que pode apontar para outra
    # instalação, sem torch/torchvision instalados).
    treino = os.path.join('Detectors', 'FasterRCNN', 'train.py')
    resultado = subprocess.run([sys.executable, treino])

    if resultado.returncode != 0:
        raise RuntimeError(
            f"O treino do Faster R-CNN falhou (código {resultado.returncode}). "
            "Veja a saída acima."
        )

    # Confere que o treino REALMENTE gerou a pasta de saida antes de tentar
    # mover -- se o train.py "engoliu" um erro fatal internamente (ele tem
    # blocos except que so imprimem e continuam) e nunca criou nada, e
    # melhor falhar aqui com uma mensagem clara do que com o FileNotFoundError
    # confuso do shutil.move.
    if not os.path.exists('Faster'):
        raise RuntimeError(
            "train.py terminou (código 0) mas não criou a pasta 'Faster' "
            "com os pesos. Confira o log acima em busca de '[ERRO FATAL]' "
            "ou '[ERRO]' repetido em todas as épocas."
        )

    # Verifica que a pasta Fold_num existe
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # shutil.move em vez de os.rename: o os.rename falha ao mover arquivos
    # entre unidades diferentes no Windows (ex.: de E:\ para C:\). Mesma
    # correção já aplicada em RunYOLOV8.py.
    #
    # 'Faster' aqui é relativo ao diretório de trabalho atual (src/), porque
    # o train.py usa OUT_DIR = './Faster' relativo a ONDE ELE RODA — e o
    # subprocess herda o cwd do processo pai (main.py, rodando em src/).
    destino = os.path.join(fold_dir, 'Faster')
    if os.path.exists(destino):
        shutil.rmtree(destino)
    shutil.move('Faster', destino)  # Move os dados dos treinos para model_checkpoints

    shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'Faster'))  # Remove as labels geradas
