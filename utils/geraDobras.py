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
parser.add_argument('-folds', default='5',dest='folds', type=int,
                    help="Número de dobras a ser usado",required=False)
parser.add_argument('-valperc', default='0.3',dest='valperc', type=float,
                    help="Percentual a ser usado para validação durante o treinamento",required=False)
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignora imagens que não tenham nenhuma anotação')
parser.add_argument('--group-by-source', dest='group_by_source', action='store_true',
                    help='Mantém todas as variantes de augmentation de uma mesma imagem-fonte '
                         'na mesma dobra. O Roboflow nomeia as variantes como '
                         '<fonte>.rf.<hash>.jpg, então a fonte é o trecho antes de ".rf.". '
                         'Sem esta flag, variantes da mesma foto caem em dobras diferentes e '
                         'o modelo é testado em cenas que já viu no treino.')
parser.add_argument('--one-variant-test', dest='one_variant_test', action='store_true',
                    help='Usa apenas UMA variante por fonte no conjunto de teste (as demais são '
                         'descartadas do teste). Evita avaliar sobre imagens augmentadas e que '
                         'a mesma cena pese N vezes na média. Requer --group-by-source.')

args = parser.parse_args()


def source_key(file_name):
    """Nome da imagem-fonte: o Roboflow gera '<fonte>.rf.<hash>.<ext>' por variante."""
    return file_name.split('.rf.')[0]


def group_images(images):
    """Agrupa as imagens por fonte, preservando a ordem de aparição."""
    grupos = {}
    for img in images:
        grupos.setdefault(source_key(img['file_name']), []).append(img)
    return grupos

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

            if args.group_by_source:
                # Particiona por IMAGEM-FONTE, não por arquivo: as variantes de
                # augmentation de uma mesma foto (mesmo prefixo antes de ".rf.")
                # vão todas para a mesma dobra. Sem isso, uma variante pode cair
                # no treino e outra no teste — o modelo seria avaliado numa cena
                # que já viu, e a métrica ficaria otimista.
                grupos = group_images(images)
                chaves = list(grupos.keys())
                qtd_teste = len(chaves)//args.folds
                print('Agrupando por imagem-fonte (.rf.)')
                print('Fontes = ', len(chaves), ' | Imagens = ', number_of_images)
                print('Quantidade de Fontes em Cada Conjunto de Teste = ', qtd_teste)

                restante = chaves
                folds_chaves = []
                for i in range(0, args.folds-1):
                    restante, z = train_test_split(restante, test_size=qtd_teste)
                    folds_chaves.append(z)
                folds_chaves.append(restante)

                # Converte os grupos de volta em listas de imagens
                folds = [funcy.lcat(grupos[k] for k in fk) for fk in folds_chaves]
            else:
                qtd_teste = number_of_images//args.folds  # Duas barras para fazer divisão inteira (sem resto)
                print('Quantidade de Imagens em Cada Conjunto de Teste = ',qtd_teste)

                # Crias as dobras
                folds_chaves = None
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

                if args.group_by_source:
                    if args.one_variant_test:
                        # Mantém só a primeira variante de cada fonte no teste: as
                        # demais são a mesma cena augmentada, então avaliá-las
                        # inflaria o peso daquela cena e mediria desempenho sobre
                        # imagens artificiais em vez de fotos reais.
                        vistas = set()
                        z_unico = []
                        for img in z:
                            chave = source_key(img['file_name'])
                            if chave not in vistas:
                                vistas.add(chave)
                                z_unico.append(img)
                        print('Teste reduzido a 1 variante por fonte: ',
                              len(z), ' -> ', len(z_unico), ' imagens')
                        z = z_unico

                    # Treino/validação também são separados por fonte, para que a
                    # validação não sirva de espelho do treino durante o early stopping.
                    grupos_xy = group_images(xy)
                    chaves_xy = list(grupos_xy.keys())
                    ch_treino, ch_val = train_test_split(chaves_xy, test_size=args.valperc)
                    x = funcy.lcat(grupos_xy[k] for k in ch_treino)
                    y = funcy.lcat(grupos_xy[k] for k in ch_val)
                else:
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

                # Confere que nenhuma fonte do teste aparece no treino/validação.
                fontes_teste = {source_key(i['file_name']) for i in z}
                fontes_treino = {source_key(i['file_name']) for i in x + y}
                vazamento = fontes_teste & fontes_treino
                if vazamento:
                    print("  [ATENCAO] {} de {} fontes do teste também estão no treino"
                          .format(len(vazamento), len(fontes_teste)))
                    if args.group_by_source:
                        raise SystemExit("Vazamento com --group-by-source: isso é um bug.")
                    print("  Use --group-by-source para eliminar esse vazamento.")
                else:
                    print("  Sem vazamento: nenhuma fonte do teste aparece no treino.")
if __name__ == "__main__":
    main(args)
