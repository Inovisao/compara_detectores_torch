import yaml
import os


def Settings():
    diretorio_de_execucao = os.getcwd()

    file_path = f"/{diretorio_de_execucao.split('/')[1]}/{diretorio_de_execucao.split('/')[2]}/.config/Ultralytics/settings.yaml"
    # O novo valor que você quer para datasets_dir

    new_datasets_dir = diretorio_de_execucao[0:-3]+'dataset'
    print(new_datasets_dir)
    input()
    # 1. Ler o arquivo YAML
    try:
        with open(file_path, 'r') as f:
            # Carrega o conteúdo do arquivo para um dicionário Python
            settings_data = yaml.safe_load(f)

        # 2. Modificar o valor no dicionário
        print(f"Valor antigo de 'datasets_dir': {settings_data.get('datasets_dir')}")
        settings_data['datasets_dir'] = new_datasets_dir
        print(f"Novo valor de 'datasets_dir': {settings_data['datasets_dir']}")

        # 3. Escrever as alterações de volta no arquivo
        with open(file_path, 'w') as f:
            # Salva o dicionário modificado de volta para o formato YAML
            yaml.dump(settings_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Arquivo '{file_path}' atualizado com sucesso!")
        input()
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{file_path}' não foi encontrado.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")