import os
import shutil
import subprocess
import sys
from Detectors.Detr.GeraDobras import convert_coco_to_voc

# Função para Rodar a rede
def runDetr(fold,fold_dir,ROOT_DATA_DIR):

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    convert_coco_to_voc(fold)

    # A versão original executava o shell script TreinoDetr.sh, que no
    # Windows falha com "WinError 193: não é um aplicativo Win32 válido"
    # (mesmo problema já corrigido em RunYOLOV8.py e runFaster.py). O .sh
    # continha apenas 'python Detectors/Detr/train_detector.py', então
    # chamamos train_detector.py diretamente com sys.executable para
    # garantir o mesmo interpretador usado pelo main.py.
    treino = os.path.join('Detectors', 'Detr', 'train_detector.py')
    resultado = subprocess.run([sys.executable, treino])

    if resultado.returncode != 0:
        raise RuntimeError(
            f"O treino do Detr falhou (código {resultado.returncode}). "
            "Veja a saída acima."
        )

    # shutil.move em vez de os.rename: o os.rename falha ao mover arquivos
    # entre unidades diferentes no Windows (ex.: de E:\ para C:\). Mesma
    # correção já aplicada em RunYOLOV8.py e runFaster.py.
    destino = os.path.join(fold_dir, "Detr")
    if os.path.exists(destino):
        shutil.rmtree(destino)
    shutil.move("./Detr", destino)
    shutil.rmtree(os.path.join(ROOT_DATA_DIR,'detr'))
