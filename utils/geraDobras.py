# Baseado no código de Artur Karaźniewicz
# Disponível aqui: https://github.com/akarazniewicz/cocosplit
#
# O número de folds padrão é 5 
# O percentual a ser usado para validação é 0.3 (30%)
# Os arquivos .json resultantes são salvos na pasta ../dataset/filesJSON
#
# Para mudar estes valores basta passar valores diferentes como parâmetros

import json
import argparse
import funcy
import os
import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Divide o conjunto de anotações para permitir aplicação de validação cruzada em dobras')
parser.add_argument('-annotations', default='../dataset/all/train/_annotations.coco.json',metavar='coco_annotations', type=str,
                    help='Caminho para o arquivo com as anotações',required=False)
parser.add_argument('-json', default='../dataset/all/filesJSON/',type=str, help='Pasta para os arquivos resultantes',required=False)
parser.add_argument('-folds', default='20',dest='folds', type=int,
                    help="Número de dobras a ser usado",required=False)
parser.add_argument('-valperc', default='0.3',dest='valperc', type=float,
                    help="Percentual a ser usado para validação durante o treinamento",required=False)
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignora imagens que não tenham nenhuma anotação')

args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def geraUma(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations_file:
        coco = json.load(annotations_file)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        # Filtra imagens que não tenham anotações, se necessário
        if args.having_annotations:
            image_ids_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
            images = funcy.lremove(lambda i: i['id'] not in image_ids_with_annotations, images)

        # Testa se a pasta para os arquivos JSON ainda não existe e cria
        if not os.path.exists(args.json):
            os.makedirs(args.json)

        # Remove os arquivos antigos da pasta filesJSON
        files = glob.glob(args.json + '*')
        for f in files:
            os.remove(f)

        # Divide entre treino, validação e teste
        images_train, images_test = train_test_split(images, test_size=0.2)
        images_train, images_val = train_test_split(images_train, test_size=args.valperc)

        # Salva os arquivos resultantes
        save_coco(os.path.join(args.json, 'fold_1_train.json'), info, licenses, images_train, filter_annotations(annotations, images_train), categories)
        save_coco(os.path.join(args.json, 'fold_1_val.json'), info, licenses, images_val, filter_annotations(annotations, images_val), categories)
        save_coco(os.path.join(args.json, 'fold_1_test.json'), info, licenses, images_test, filter_annotations(annotations, images_test), categories)

        print("Salvou {} anotações em {}".format(len(images_train), 'fold_1_train.json'))
        print("Salvou {} anotações em {}".format(len(images_val), 'fold_1_val.json'))
        print("Salvou {} anotações em {}".format(len(images_test), 'fold_1_test.json'))


def main(args):
    if args.folds == 1:
        geraUma(args)
    else:
        with open(args.annotations, 'rt', encoding='UTF-8') as annotations:

            coco = json.load(annotations)
            info = coco['info']
            licenses = coco['licenses']
            images = coco['images']
            annotations = coco['annotations']
            categories = coco['categories']

            number_of_images = len(images)

            images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

            if args.having_annotations:
                images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

            # Testa se a pasta para os arquivos JSON ainda não existe e cria
            if not os.path.exists(args.json):
                os.makedirs(args.json)
                
            # Remove os arquivos antigos da pasta filesJSON
            files = glob.glob(args.json+'*')
            for f in files:
                os.remove(f)

            qtd_teste = number_of_images//args.folds  # Duas barras para fazer divisão inteira (sem resto)
            print('Quantidade de Imagens em Cada Conjunto de Teste = ',qtd_teste)

            # Crias as dobras 
            folds=[]
            for i in range(0,args.folds-1):
                images, z = train_test_split(images, test_size=qtd_teste)
                folds.append(z)
            folds.append(images)
            
            for i in range(0,args.folds):

                print('---------------------------')
                print('Processando Dobra ',i+1)
                z=folds[i] # Conjunto de teste para a dobra i
                xy=[]  # Vai juntar as outras dobras aqui

                for j in range(0, args.folds):
                    if i!=j:
                        xy=xy+folds[j]

                # Aqui xy está sendo dividido entre treino e validação
                x, y = train_test_split(xy, test_size=args.valperc)

                arq_treino=args.json+'fold_'+str(i+1)+'_train.json'
                arq_val=args.json+'fold_'+str(i+1)+'_val.json'
                arq_teste=args.json+'fold_'+str(i+1)+'_test.json'
                
                save_coco(arq_treino, info, licenses, x, filter_annotations(annotations, x), categories)
                save_coco(arq_val, info, licenses, y, filter_annotations(annotations, y), categories)
                save_coco(arq_teste, info, licenses, z, filter_annotations(annotations, z), categories)

                print("Salvou {} anotações em {}".format(len(x), arq_treino))
                print("Salvou {} anotações em {}".format(len(y), arq_val))
                print("Salvou {} anotações em {}".format(len(z), arq_teste))
if __name__ == "__main__":
    main(args)
