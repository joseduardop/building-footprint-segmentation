# Segmentação de Edificações em Imagens de Satélite com U-Net e ResNet34

**José Eduardo Pereira** - EEL7815 Processamento Digital de Imagens - UFSC, 2026.1
Prof. Joceli Mayer

---

## Resumo

Este trabalho apresenta uma pipeline completa para segmentação de edificações em
imagens de satélite e extração de polígonos georreferenciados. A solução combina
técnicas de Processamento Digital de Imagens (realce radiométrico, limiarização e
morfologia matemática) com uma rede neural U-Net de backbone ResNet34 pré-treinado.
Foram comparadas duas estratégias de tratamento da resolução de entrada - *tiling*
e *resize* -, com a primeira alcançando F1-score de 0,62 e IoU médio de 0,72 sobre
o conjunto de teste do dataset SpaceNet 2 (Paris).

## 1. Introdução

O reconhecimento automático de edificações em imagens aéreas é um problema central
em sensoriamento remoto, com aplicações em cartografia, planejamento urbano e
resposta a desastres. Trata-se de uma tarefa de segmentação semântica: classificar
cada pixel como pertencente a uma edificação ou ao fundo. O objetivo deste projeto
é, dada uma imagem de satélite de um recorte urbano, identificar as edificações e
produzir polígonos vetoriais em coordenadas geográficas (latitude e longitude),
diretamente utilizáveis em ferramentas de SIG como o QGIS.

## 2. Materiais e Métodos

Utilizou-se o dataset **SpaceNet 2, AOI 3 (Paris)**: 1148 tiles de 650×650 pixels,
16 bits, 3 bandas (RGB-PanSharpen), com anotações vetoriais (GeoJSON) das
edificações. Os dados foram divididos em 70% treino, 15% validação e 15% teste, de
forma reprodutível.

O **pré-processamento** aplica normalização por percentil (2–98), uma transformação
radiométrica que expande o contraste das imagens de 16 bits. A **arquitetura** é uma
U-Net com codificador ResNet34 pré-treinado no ImageNet (*transfer learning*),
treinada com função de perda combinada BCE + Dice e otimizador Adam
(*learning rate* 1e-4), com *early stopping* e redução de *learning rate* em platô.

O **pós-processamento** converte a máscara de probabilidade em polígonos: limiarização
(0,5), morfologia matemática (abertura seguida de fechamento), extração e
simplificação de contornos, e conversão de coordenadas de pixel para latitude/longitude.
Foram comparadas duas estratégias de entrada: **tiling** (recorte do tile em janelas de
256×256 na resolução original, preservando detalhe) e **resize** (redimensionamento do
tile inteiro para 256×256, mais simples porém com perda de detalhe).

## 3. Resultados

A avaliação é feita no nível de instância (por edificação): uma predição é
considerada correta quando há um rótulo correspondente com IoU > 0,5.

| Métrica   | Tiling | Resize |
|-----------|:------:|:------:|
| Precisão  | 0,70   | 0,68   |
| Revocação | 0,55   | 0,47   |
| F1-score  | 0,62   | 0,56   |
| IoU médio | 0,72   | 0,70   |

A estratégia **tiling superou a resize em todas as métricas**, com diferença mais
marcante na revocação. A causa é direta: o redimensionamento comprime o tile e
suprime as edificações menores, que deixam de ser detectadas; o tiling, ao preservar
a resolução original, mantém o detalhe e encontra mais construções. Em teste de
generalização com imagens de satélite externas ao dataset, o modelo detectou
edificações de forma consistente, desde que a escala aparente das construções fosse
compatível com a do treinamento (cerca de 0,3 m/pixel), evidenciando sensibilidade
à escala.

## 4. Conclusão

A pipeline cumpre o objetivo de segmentar edificações e gerar polígonos
georreferenciados, integrando realce, limiarização e morfologia a uma U-Net com
*transfer learning*. O *tiling* mostrou-se a abordagem mais adequada para o dataset.
Como trabalhos futuros, destacam-se o treino com múltiplas escalas (para reduzir a
sensibilidade ao *zoom*), a limiarização adaptativa (Otsu) e a avaliação em
múltiplos limiares de IoU.

## Referências

RONNEBERGER, O.; FISCHER, P.; BROX, T. *U-Net: Convolutional Networks for
Biomedical Image Segmentation*. MICCAI, 2015. • HE, K. et al. *Deep Residual
Learning for Image Recognition*. CVPR, 2016. • VAN ETTEN, A. et al. *SpaceNet: A
Remote Sensing Dataset and Challenge Series*. 2018. • GONZALEZ, R. C.; WOODS, R. E.
*Digital Image Processing*. Pearson.

SpaceNet Challenge. Disponível em: <https://github.com/spacenetchallenge>.
SpaceNet Datasets. Disponível em: <https://spacenet.ai/datasets/>.
SpaceNet Buildings Dataset v2. AWS Registry of Open Data. Disponível em:
<https://registry.opendata.aws/spacenet/>.
SpaceNet Buildings Dataset v2. Disponível em:
<https://spacenet.ai/spacenet-buildings-dataset-v2/>.
