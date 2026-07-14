import os
import subprocess
import sys
import shutil
from pathlib import Path

from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from Detectors.YOLOV8.TrocaSettings import Settings


def _localizar_saida():
    """Descobre onde o Ultralytics gravou o resultado do treino.

    O config.py passa project='YOLOV8'. O caminho final mudou entre versões:

      Ultralytics ~8.2 (a fixada no README):  src/YOLOV8/
      Ultralytics ~8.4 (atual):               src/runs/detect/YOLOV8/

    Em vez de fixar um dos dois, procuramos nos dois. Assim o código funciona
    tanto com a versão antiga quanto com a nova.
    """
    candidatos = [
        Path('YOLOV8'),
        Path('runs') / 'detect' / 'YOLOV8',
    ]
    for c in candidatos:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Não encontrei a pasta de saída do treino da YOLOV8. Procurei em: "
        f"{[str(c) for c in candidatos]}. Confira a linha 'Results saved to ...' "
        "no log do Ultralytics e ajuste _localizar_saida()."
    )


def runYOLOV8(fold, fold_dir, ROOT_DATA_DIR):

    Settings()
    CriarLabelsYOLOV8(fold)  # Cria as labels no formato YOLO para esta dobra

    # A versão original executava o shell script TreinoYOLOV8.sh, que no Windows
    # falha com "WinError 193: não é um aplicativo Win32 válido". O .sh continha
    # apenas uma linha útil ('python Detectors/YOLOV8/config.py'), então
    # chamamos o config.py diretamente — funciona em Windows, Linux e macOS.
    #
    # sys.executable garante o MESMO interpretador que está rodando o main.py.
    # O .sh chamava 'python', que pode apontar para outra instalação, sem torch.
    config = os.path.join('Detectors', 'YOLOV8', 'config.py')
    resultado = subprocess.run([sys.executable, config])

    if resultado.returncode != 0:
        raise RuntimeError(
            f"O treino da YOLOV8 falhou (código {resultado.returncode}). "
            "Veja a saída acima."
        )

    origem = _localizar_saida()

    destino_fold = Path(fold_dir)
    destino_fold.mkdir(parents=True, exist_ok=True)

    destino = destino_fold / 'YOLOV8'
    if destino.exists():
        shutil.rmtree(destino)

    # shutil.move em vez de os.rename: o os.rename falha ao mover arquivos
    # entre unidades diferentes no Windows (ex.: de E:\ para C:\).
    shutil.move(str(origem), str(destino))

    # Remove a pasta 'runs' que o Ultralytics novo cria, se tiver ficado vazia.
    runs = Path('runs')
    if runs.is_dir() and not any(runs.rglob('*')):
        shutil.rmtree(runs, ignore_errors=True)

    # Remove as labels temporárias geradas para esta dobra.
    shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'YOLO'), ignore_errors=True)
