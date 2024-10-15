import os
import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil 
import random
import argparse

def format2od(input_dir, output_positive_dir, output_negative_dir, model):
        # Cria os diretórios de saída, se não existirem
        os.makedirs(output_positive_dir, exist_ok=True)
        os.makedirs(output_negative_dir, exist_ok=True)
        # Lista todos os arquivos no diretório de entrada
        for filename in os.listdir(input_dir):
            # Verifica se o arquivo é uma imagem
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_dir, filename)
                results = model.predict(image_path)
                # Carrega a imagem original
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB para exibição
                # Checa se há detecções
                detections_exist = any(len(result.boxes) > 0 for result in results)
                if detections_exist:
                    # Salva a imagem na pasta "positive"
                    output_image_path = os.path.join(output_positive_dir, filename)
                    cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Converte RGB de volta para BGR para salvar
                    print(f'Saved positive detection: {output_image_path}')
                    # Salva as detecções em um arquivo .txt
                    height, width, _ = image.shape
                    output_txt_path = os.path.join(output_positive_dir, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                    with open(output_txt_path, 'w') as f:
                        for result in results:
                            boxes = result.boxes  # Obtém as caixas do resultado
                            for box in boxes:
                                class_id = int(box.cls[0].cpu().item())  # Obtém a classe (como inteiro)
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Obtém as coordenadas da caixa
                                normalized_x1 = x1 / width
                                normalized_y1 = y1 / width
                                normalized_x2 = x2 / width
                                normalized_y2 = y2 / width
                                # Escreve a classe e as coordenadas no arquivo .txt
                                f.write(f'{class_id} {normalized_x1} {normalized_y1} {normalized_x2} {normalized_y2}\n')
                else:
                    # Salva a imagem na pasta "negative"
                    output_path = os.path.join(output_negative_dir, filename)
                    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Converte RGB de volta para BGR para salvar
                    print(f'Saved negative detection: {output_path}')

def format2seg(input_dir, output_positive_dir, output_negative_dir, model):
    # Cria os diretórios de saída, se não existirem
    os.makedirs(output_positive_dir, exist_ok=True)
    os.makedirs(output_negative_dir, exist_ok=True)

    # Lista todos os arquivos no diretório de entrada
    for filename in os.listdir(input_dir):
        # Verifica se o arquivo é uma imagem
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            results = model.predict(image_path)

            # Carrega a imagem original
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB para exibição

            # Checa se há detecções
            detections_exist = any(len(result.boxes) > 0 for result in results)

            if detections_exist:
                # Salva a imagem na pasta "positive"
                output_image_path = os.path.join(output_positive_dir, filename)
                cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Converte RGB de volta para BGR para salvar
                print(f'Saved positive detection: {output_image_path}')

                # Salva as detecções em um arquivo .txt
                height, width, _ = image.shape
                output_txt_path = os.path.join(output_positive_dir, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                
                with open(output_txt_path, 'w') as f:
                    for result in results:
                        boxes = result.boxes  # Obtém as caixas do resultado
                        masks = result.masks  # Obtém as máscaras de segmentação
                        
                        for i in range(len(boxes)):
                            class_id = int(boxes[i].cls[0].cpu().item())  # Obtém a classe (como inteiro)

                            # Obtenha a máscara de segmentação
                            mask = masks[i]  # Isso é um objeto de máscara

                            # Obtenha a máscara como coordenadas
                            mask_data = mask.xy  # Obtém as coordenadas da máscara

                            # Gera uma máscara binária
                            mask_binary = np.zeros((height, width), dtype=np.uint8)
                            for contour in mask_data:
                                cv2.fillPoly(mask_binary, [contour.astype(np.int32)], 1)

                            # Extraí os contornos da máscara binária
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            for contour in contours:
                                # Normaliza as coordenadas dos pontos do contorno
                                normalized_contour = []
                                for point in contour:
                                    normalized_x = point[0][0] / width
                                    normalized_y = point[0][1] / height
                                    normalized_contour.append((normalized_x, normalized_y))

                                # Escreve a classe e os pontos no arquivo .txt
                                if len(normalized_contour) >= 3:  # Certifique-se de que há pelo menos 3 pontos
                                    f.write(f'{class_id} ' + ' '.join(f'{x} {y}' for x, y in normalized_contour) + '\n')

            else:
                # Salva a imagem na pasta "negative"
                output_path = os.path.join(output_negative_dir, filename)
                cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Converte RGB de volta para BGR para salvar
                print(f'Saved negative detection: {output_path}')

def split_dataset(positive_dir, train_dir, val_dir, train_ratio=0.7):
        # Cria os diretórios de treino e validação, se não existirem
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Lista todos os arquivos no diretório positivo
        images = [f for f in os.listdir(positive_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Embaralha a lista de imagens
        random.shuffle(images)

        # Calcula o número de imagens para o conjunto de treino
        total_images = len(images)
        train_size = int(total_images * train_ratio)

        # Divide as imagens em treino e validação
        train_images = images[:train_size]
        val_images = images[train_size:]

        # Move as imagens e arquivos .txt para os diretórios apropriados
        for image in train_images:
            # Mover imagem
            src_image_path = os.path.join(positive_dir, image)
            dst_image_path = os.path.join(train_dir, image)
            shutil.move(src_image_path, dst_image_path)

            # Mover arquivo .txt correspondente, se existir
            txt_filename = os.path.splitext(image)[0] + '.txt'
            src_txt_path = os.path.join(positive_dir, txt_filename)
            if os.path.exists(src_txt_path):
                dst_txt_path = os.path.join(train_dir, txt_filename)
                shutil.move(src_txt_path, dst_txt_path)

        for image in val_images:
            # Mover imagem
            src_image_path = os.path.join(positive_dir, image)
            dst_image_path = os.path.join(val_dir, image)
            shutil.move(src_image_path, dst_image_path)

            # Mover arquivo .txt correspondente, se existir
            txt_filename = os.path.splitext(image)[0] + '.txt'
            src_txt_path = os.path.join(positive_dir, txt_filename)
            if os.path.exists(src_txt_path):
                dst_txt_path = os.path.join(val_dir, txt_filename)
                shutil.move(src_txt_path, dst_txt_path)

        print(f"Moved {len(train_images)} images to '{train_dir}' and {len(val_images)} images to '{val_dir}'.")
    
def train(model_path: str, dataset_yaml: str, device: str = 'cuda', epochs: int = 100, imgsz: int = 640):
    model = YOLO(model=model_path)
    model.train(data=dataset_yaml, device=device, epochs=epochs, imgsz=imgsz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python package to automate object detection and segmentation training for the YOLO v8 neural network")
    
    # Definindo subcomandos
    subparsers = parser.add_subparsers(dest='command')

    # Subcomando 'format'
    format_parser = subparsers.add_parser('format', help='Command to format the dataset for object detection or segmentation')
    format_parser.add_argument('format_type', choices=['object_detection', 'segmentation'], help='Format type: object detection or segmentation')
    format_parser.add_argument('--input_dir', type=str, help='Image directory')
    format_parser.add_argument('--output_positive_dir', type=str, help='Output positive image directory')
    format_parser.add_argument('--output_negative_dir', type=str, help='Output negative image directory')
    format_parser.add_argument('--model', type=str, help='Base model')

    # Subcomando 'split_dataset'
    split_parser = subparsers.add_parser('split_dataset', help="Command to split the dataset into the COCO format")
    split_parser.add_argument('--output_positive_dir', type=str, help='Output positive image directory')
    split_parser.add_argument('--train_dir', type=str, help='Directory for training images', default="train")
    split_parser.add_argument('--val_dir', type=str, help='Directory for validation images', default="val")
    split_parser.add_argument('--train_ratio', type=float, help='Ratio to split the dataset into train and validation. Example: 0.7 = 70% to train and 30% to validation', default=0.7, required=False)

    # Subcomando 'train'
    train_parser = subparsers.add_parser('train', help='Command to start the train')
    train_parser.add_argument('--model', type=str, help='Location of the base model')
    train_parser.add_argument('--dataset_yaml', type=str, help='Location of .yaml for train/validation dataset')
    train_parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--imgsz', type=int, default=640)

    args = parser.parse_args()

    if args.command == 'format':
        if args.format_type == 'object_detection':
            format2od(input_dir=args.input_dir, output_positive_dir=args.output_positive_dir, output_negative_dir=args.output_negative_dir, model=args.model)
        elif args.format_type == 'segmentation':
            format2seg(input_dir=args.input_dir, output_positive_dir=args.output_positive_dir, output_negative_dir=args.output_negative_dir, model=args.model)
    
    elif args.command == 'split_dataset':
        split_dataset(positive_dir=args.output_positive_dir, train_dir=args.train_dir, val_dir=args.val_dir, train_ratio=args.train_ratio)

    elif args.command == 'train':
        train(model_path=args.model, dataset_yaml=args.dataset_yaml, device=args.device, epochs=args.epochs, imgsz=args.imgsz)