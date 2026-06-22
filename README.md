# Segmentação de Edificações em Imagens de Satélite — SpaceNet 2 (AOI 3, Paris)

## 1. Disciplina

EEL7815 — Processamento Digital de Imagens
Universidade Federal de Santa Catarina (UFSC), 2026.1
Prof. Joceli Mayer
Aluno: José Eduardo Pereira

## 2. Objetivo Geral e Ideia do Projeto

O projeto implementa uma pipeline completa de segmentação semântica de
edificações em imagens de satélite de alta resolução. Dada uma imagem aérea
de um recorte urbano (tile), o sistema identifica automaticamente as
edificações e produz, como saída, polígonos vetoriais georreferenciados
(latitude/longitude) delimitando cada construção detectada.

A ideia central é tratar o problema como uma tarefa de segmentação binária
pixel-a-pixel (edificação vs. fundo), seguida de um pós-processamento que
converte a máscara de probabilidade em geometrias vetoriais utilizáveis em
ferramentas de Sistema de Informação Geográfica (SIG), como o QGIS. O produto
final é análogo ao Microsoft Global ML Building Footprints.

## 3. Objetivos Específicos

3.1. Construir um carregador de dados que paireie imagens GeoTIFF de 16 bits
com suas anotações vetoriais (GeoJSON), tratando normalização e rasterização.

3.2. Implementar e treinar uma rede U-Net com backbone ResNet34 pré-treinado,
utilizando função de perda combinada (BCE + Dice).

3.3. Aplicar técnicas de Processamento Digital de Imagens no pós-processamento:
limiarização, morfologia matemática, extração e simplificação de contornos.

3.4. Avaliar quantitativamente o modelo por métricas no nível de instância
(IoU e F1-score) e qualitativamente por sobreposição visual das predições.

3.5. Comparar duas estratégias de tratamento da resolução de entrada
(tiling vs. redimensionamento) sob o mesmo conjunto de dados.

## 4. Requisitos e Dependências

### 4.1. Requisitos Funcionais

- Validação de integridade e rasterização das anotações do dataset
- Divisão reprodutível dos dados em treino, validação e teste (70/15/15)
- Treinamento com early stopping e seleção do melhor modelo por IoU
- Inferência georreferenciada com exportação para GeoJSON (EPSG:4326)
- Avaliação por instância (correspondência quando IoU > 0,5)

### 4.2. Requisitos Técnicos (requirements.txt)

- Python 3.10+, PyTorch não se aplica — base em TensorFlow 2.16+ / tf-keras
- segmentation-models (arquitetura U-Net + backbone)
- rasterio, geopandas, shapely (dados geoespaciais)
- opencv-python-headless (morfologia, contornos)
- albumentations (data augmentation)
- matplotlib, numpy, Pillow

### 4.3. Ambiente de Execução

- GPU NVIDIA com CUDA. No Windows, o uso de GPU exige WSL2 (Ubuntu), pois o
  TensorFlow 2.16+ não suporta GPU nativamente no Windows.
- Treinamento realizado em RTX 3060 Ti (8 GB), via WSL2.

## 5. Organização do Repositório

```
p2/
├── scripts/          núcleo da pipeline
│   ├── dataset.py        carregamento de dados, splits, augmentation
│   ├── train.py          treinamento da U-Net
│   ├── postprocess.py    inferência e geração de polígonos
│   ├── evaluate.py       métricas IoU e F1 por instância
│   └── validate_dataset.py  validação e rasterização de debug
├── util/             utilitários
│   ├── create_masks.py     cache de máscaras rasterizadas
│   ├── create_previews.py  conversão de GeoTIFF 16-bit em PNG visível
│   └── visualize.py        geração de figuras
├── notebooks/        análise e demonstração
│   ├── dataset_check.ipynb  exploração do dataset
│   └── demo.ipynb           inferência interativa
├── artefatos/        modelos treinados, métricas e splits
├── figures/          figuras geradas
├── requirements.txt
└── README.md
```

## 6. Pipeline e Fluxo de Execução

A pipeline é linear; cada etapa é um script independente acionável por linha
de comando (argparse), sem caminhos fixos no código.

```mermaid
flowchart TD
    A["Dataset SpaceNet 2<br/>.tif (imagem) + .geojson (rótulo)"] --> B["create_masks.py<br/>rasteriza rótulos em máscaras"]
    B --> C["dataset.py<br/>pares (imagem, máscara)<br/>split 70/15/15"]
    C --> D["train.py<br/>U-Net ResNet34<br/>perda BCE + Dice"]
    D --> E["best_model.keras"]
    E --> F["postprocess.py<br/>inferência + morfologia<br/>+ vetorização"]
    F --> G["predictions.geojson<br/>(EPSG:4326)"]
    G --> H["evaluate.py<br/>IoU + F1 por instância"]
    H --> I["results.json"]
    E --> J["visualize.py<br/>curvas e overlays"]
```

Ordem de execução: `create_masks.py` → `train.py` → `postprocess.py` →
`evaluate.py` → `visualize.py`. O módulo `dataset.py` é importado pelo
treinamento, não executado isoladamente.

### 6.1. Arquitetura do Modelo

- Codificador: ResNet34 pré-treinado em ImageNet (transfer learning)
- Decodificador: U-Net com conexões de salto (skip connections)
- Entrada: 256×256×3; Saída: 256×256×1 (sigmoide)
- Perda: BCE + Dice (0,5 / 0,5); Otimizador: Adam (lr = 1e-4)
- Normalização: percentil 2–98 por tile (16 bits → [0, 1])

## 7. Resultados e Testes

Dataset: 1148 tiles rotulados (650×650, 16 bits), divididos em 803 treino /
172 validação / 173 teste. Avaliação no conjunto de teste interno.

Foram treinadas duas configurações sob o mesmo split e mesma semente:

| Métrica          | Tiling      | Resize      |
|------------------|-------------|-------------|
| IoU de validação | **0,7251**  | 0,6327      |
| Precisão         | **0,70**    | 0,68        |
| Revocação        | **0,55**    | 0,47        |
| F1-score         | **0,62**    | 0,56        |
| IoU médio        | **0,72**    | 0,70        |
| Épocas           | 37          | 40          |

A configuração *tiling* (recorte em resolução nativa) superou o
*resize* (redimensionamento do tile inteiro), principalmente em revocação:
o redimensionamento comprime os tiles e suprime edificações pequenas,
reduzindo o número de detecções corretas. Ambos os treinamentos ocorreram em
GPU, com duração aproximada de 30 minutos cada.

Validação qualitativa: o modelo treinado generalizou para imagens fora do
domínio (capturas do Google Maps de Florianópolis), detectando edificações
desde que a escala aparente fosse compatível com a do treinamento.

## 8. Próximos Passos

- Substituir o limiar fixo (0,5) por limiarização adaptativa (Otsu)
- Ajustar área mínima e elemento estruturante da morfologia ao domínio
- Explorar fusão das modalidades multiespectrais (MUL) do SpaceNet
- Aplicar o modelo ao conjunto de teste oficial (381 tiles sem rótulo) com
  avaliação qualitativa em ferramenta SIG
