from pathlib import Path

from ultralytics import settings


def Settings():
    """Aponta o `datasets_dir` do Ultralytics para a pasta `dataset` do projeto.

    A versão anterior montava o caminho do arquivo de configuração na mão:

        f"/{cwd.split('/')[1]}/{cwd.split('/')[2]}/.config/Ultralytics/settings.yaml"

    Isso só funciona no Linux. No Windows o separador é '\\', então o split('/')
    devolve uma lista de um elemento e o índice [1] estoura (IndexError). Além
    disso, no Windows a configuração fica em AppData (e nas versões recentes do
    Ultralytics o arquivo é settings.json, não .yaml) — ou seja, nem o caminho
    nem o formato batiam.

    A API `ultralytics.settings` resolve os dois problemas: ela sabe onde fica a
    configuração em cada sistema operacional e cuida do formato do arquivo.
    """
    # Este arquivo está em <raiz>/src/Detectors/YOLOV8/TrocaSettings.py.
    # parents[3] sobe quatro níveis e chega na raiz do projeto — assim o
    # caminho não depende de onde o script foi executado.
    raiz = Path(__file__).resolve().parents[3]
    datasets_dir = raiz / "dataset"

    print(f"Valor antigo de 'datasets_dir': {settings.get('datasets_dir')}")
    settings.update({"datasets_dir": str(datasets_dir)})
    print(f"Novo valor de 'datasets_dir': {settings.get('datasets_dir')}")
