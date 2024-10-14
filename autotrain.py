from ultralytics import YOLO
import cv2
import os
import shutil 
import random

class Autotrain:
    def detect_and_save_images(input_dir, output_positive_dir, output_negative_dir, model):
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
    
    def train(model_path: str, dataset_yaml: str, device: str = 'cuda', epochs: int = 100, imgsz: int = 620):
        model = YOLO(model=model_path)
        model.train(data=dataset_yaml, device=device, epochs=epochs, imgsz=imgsz)