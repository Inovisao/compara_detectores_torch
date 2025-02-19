import cv2
import os

def resize_image(input_path, output_path, size=(640, 640)):
    # Ler a imagem usando OpenCV
    img = cv2.imread(input_path)
    if img is not None:
        # Redimensionar a imagem
        img_resized = cv2.resize(img, size)
        # Salvar a imagem redimensionada
        cv2.imwrite(output_path, img_resized)
    else:
        print(f"Erro ao abrir a imagem: {input_path}")

def resize_images_in_directory(input_dir, output_dir, size=(640, 640)):
    # Certificar-se de que o diretório de saída exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Percorrer todos os arquivos no diretório de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            # Redimensionar cada imagem
            resize_image(input_path, output_path, size)

# Exemplo de uso
input_directory = 'train'  # Diretório de entrada
output_directory = '../dataset/all/train'  # Diretório de saída
resize_images_in_directory(input_directory, output_directory)
