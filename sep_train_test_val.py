import os
import shutil
import random
from pathlib import Path

#-----------------------------------------------------------------------------------------------------------
# Este script separa as imagens em pastas de treino, teste e validação
# com base na proporção especificada. As imagens são distribuídas
# aleatoriamente entre as pastas de destino.
# Definindo o caminho da pasta original e as classes
#-----------------------------------------------------------------------------------------------------------

# Caminho da pasta original com as classes
origem = Path("/home/carlos/workspace/synapse/mechanical_parts_classifier/mecanic_parts/archive/dataset/training")
classes = ['arruela', 'parafuso', 'pinoGuia', 'parafusoVazado']
destinos = {'train': 0.7, 'test': 0.2, 'val': 0.1}

# Criar diretórios de destino (train, test, val) com subpastas para cada classe
for destino in destinos:
    for classe in classes:
        # Criar diretório de destino e subdiretório para cada classe
        Path(destino, classe).mkdir(parents=True, exist_ok=True)

# Função para distribuir os arquivos
def distribuir_imagens():
    # Itera sobre cada classe
    for classe in classes:
        # Caminho da classe na pasta original 
        caminho_classe = origem / classe
        # imagens recebe a lista de todas as imagens na pasta da classe
        imagens = list(caminho_classe.glob("*.png"))
        #separa randomicamente as imagens
        random.shuffle(imagens)
        # Verifica se há imagens suficientes para a divisão
        total = len(imagens)
        # train_count, test_count e val_count são os números de imagens para cada conjunto
        train_count = int(destinos['train'] * total)
        test_count = int(destinos['test'] * total)
        val_count = total - train_count - test_count

        # Faz a divisão das imagens
        divisao = {
            # train : 70% das imagens
            #70% : imagens[:train_count],
            'train': imagens[:train_count],
            # test : 20% das imagens
            #90% : imagens[train_count:train_count+test_count],
            'test': imagens[train_count:train_count+test_count],
            # val : 10% das imagens
            # Depois de 90% : imagens[train_count+test_count:]
            'val': imagens[train_count+test_count:]
        }

        # Copiar imagens para as pastas de destino
        for tipo_destino, lista_imagens in divisao.items():
            for imagem in lista_imagens:
                destino_final = Path(tipo_destino, classe, imagem.name)
                shutil.copy2(imagem, destino_final)

# Executar a função
if __name__ == "__main__":
    distribuir_imagens()
    print("Separação concluída com sucesso.")
