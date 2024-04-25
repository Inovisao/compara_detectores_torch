import os
import re

# Função para pegar o nome do checkpoint baixado
def Checkpoint(model):
    diretorio = 'Detectors/MMdetection/checkpoints'
    # Padrão de expressão regular para fazer correspondência parcial no nome do arquivo
    padrao = re.compile(fr'^{model}.*\.pth$')
    # Lista para armazenar os caminhos dos arquivos com nomes semelhantes
    arquivos_com_nomes_semelhantes = []
    # Itera sobre todos os arquivos no diretório
    for nome in os.listdir(diretorio):
        caminho_completo = os.path.join(diretorio, nome)
        if os.path.isfile(caminho_completo) and padrao.match(nome):
            arquivos_com_nomes_semelhantes.append(caminho_completo)
    return arquivos_com_nomes_semelhantes[0]

# Função para pegar o ultimo modelo Treinado
def selectmodel(fold,model):
    diretorio = 'model_checkpoints/'+fold+'/'+model
    # Padrão de expressão regular para fazer correspondência parcial no nome do arquivo
    padrao = re.compile(fr'^epoch_.*\.pth$')
    # Lista para armazenar os caminhos dos arquivos com nomes semelhantes

    arquivos_com_nomes_semelhantes = []
    # Itera sobre todos os arquivos no diretório
    for nome in os.listdir(diretorio):
        caminho_completo = os.path.join(diretorio, nome)
        if os.path.isfile(caminho_completo) and padrao.match(nome):
            arquivos_com_nomes_semelhantes.append(caminho_completo)

    return arquivos_com_nomes_semelhantes[1].split('/')[-1]