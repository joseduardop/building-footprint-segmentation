# segmentacao de edificacoes - spacenet 2 paris

pipeline de segmentacao semantica de edificacoes em imagens de satelite,
treinada no dataset spacenet 2 (aoi 3, paris). o modelo recebe tiles
rgb-pansharpen 650x650 16-bit e produz poligonos georreferenciados
(lat/lon) de cada edificacao detectada.

projeto da disciplina EEL7815 - processamento digital de imagens
(UFSC, prof. joceli mayer, 2026.1).

## resultados

| metrica | valor |
|---------|-------|
| precision | 0.70 |
| recall | 0.55 |
| F1-score | 0.62 |
| IoU medio | 0.72 |
| epocas treinadas | 37 (early stopping) |
| melhor val IoU | 0.7251 (epoca 27) |

treinado em GPU (RTX 3060 Ti, ~30 min). modo tiling com overlap 50%.

## requisitos

- python 3.10+
- tensorflow >= 2.16
- GPU nvidia com CUDA (via WSL2 no windows)
- dataset spacenet 2 aoi_3_paris (~10GB, nao incluido no repo)
- dependencias em `requirements.txt`

```bash
pip install -r requirements.txt
```

pra GPU no windows, rode dentro do WSL2 com o venv configurado
(ver secao "ambiente" abaixo).

## estrutura do dataset

o dataset deve estar disponivel localmente nesta estrutura:

```
spacenet/
  train/
    RGB-PanSharpen/    (1148 tiles .tif)
    geojson/buildings/ (1148 .geojson - labels)
    masks/             (gerado pelo create_masks.py)
  test/
    RGB-PanSharpen/    (381 tiles .tif, SEM labels)
```

tiles: 650x650 pixels, 16-bit, 3 bandas (rgb-pansharpen).
labels: poligonos de edificacoes em geojson (CRS84 = lon/lat).
muitos tiles tem zero edificacoes - e valido, o modelo aprende ausencia.

## estrutura do projeto

```
p2/
  scripts/
    dataset.py          # carregador de dados + splits 70/15/15
    train.py            # treino da u-net resnet34
    postprocess.py      # inferencia tiling -> poligonos geojson
    evaluate.py         # metricas IoU + F1 por instancia
    validate_dataset.py # validacao e debug do dataset
  util/
    create_masks.py     # cache de mascaras (roda 1x antes do treino)
    create_previews.py  # converte tif 16-bit em png visualizavel
    visualize.py        # gera figuras (overlays, curvas, grid)
  notebooks/
    demo.ipynb          # inferencia interativa (tile ou google maps)
  artefatos/
    best_model.keras    # modelo treinado (294MB, fora do git)
    predictions.geojson # poligonos detectados no test split
    results.json        # metricas finais
    splits.json         # divisao train/val/test (reprodutivel)
    training_history.json # loss e IoU por epoca
  figures/              # figuras geradas pelo visualize.py
```

## pipeline (ordem de execucao)

### 1. cache de mascaras (roda uma vez)

pre-rasteriza os geojson em mascaras png pra acelerar o treino.

```bash
python util/create_masks.py --data_dir /caminho/spacenet
```

### 2. treino

```bash
python scripts/train.py \
    --data_dir /caminho/spacenet \
    --output_dir artefatos \
    --mode tiling \
    --epochs 50 \
    --batch_size 8
```

modos disponiveis:
- `--mode tiling` : crops 256x256 na resolucao original (preserva detalhe)
- `--mode resize` : tile inteiro redimensionado pra 256 (sem emendas)

saida: `best_model.keras`, `training_history.json`, `splits.json`

### 3. pos-processamento

inferencia nos tiles do test split, gerando poligonos georreferenciados.

```bash
python scripts/postprocess.py \
    --data_dir /caminho/spacenet/train/RGB-PanSharpen \
    --model artefatos/best_model.keras \
    --output artefatos/predictions.geojson \
    --splits_json artefatos/splits.json \
    --split test \
    --mode tiling
```

### 4. avaliacao

calcula IoU e F1 por instancia (match quando IoU > 0.5).

```bash
python scripts/evaluate.py \
    --predictions artefatos/predictions.geojson \
    --ground_truth /caminho/spacenet/train/geojson/buildings/ \
    --splits_json artefatos/splits.json \
    --split test \
    --output artefatos/results.json
```

### 5. figuras

```bash
# overlays predicao vs ground truth
python util/visualize.py overlay \
    --data_dir /caminho/spacenet \
    --predictions artefatos/predictions.geojson \
    --splits_json artefatos/splits.json \
    --split test

# curvas de treino (loss e IoU)
python util/visualize.py curves --history artefatos/training_history.json

# grid de amostras do dataset
python util/visualize.py grid --data_dir /caminho/spacenet
```

### utilitarios

```bash
# previews png dos tiles 16-bit (pra abrir no visualizador normal)
python util/create_previews.py --data_dir /caminho/spacenet

# demo interativo (notebook)
jupyter notebook notebooks/demo.ipynb
```

## arquitetura do modelo

- encoder: resnet34 pre-treinado no imagenet (transfer learning)
- decoder: u-net com skip connections
- loss: BCE + dice (0.5 / 0.5)
- otimizador: adam, lr=1e-4
- callbacks: early stopping (patience=10), checkpoint (val IoU), reduce lr
- entrada: 256x256x3 (crop aleatorio do tile 650x650)
- saida: 256x256x1 (mascara binaria, sigmoid)
- normalizacao: percentil 2-98 por tile (16-bit -> [0,1])
- pos-processamento: limiar 0.5 -> morfologia (abertura+fechamento) ->
  contornos -> simplificacao -> filtro de area -> georreferenciamento

## ambiente (GPU via WSL2)

no windows, tensorflow so usa GPU pelo WSL2:

```bash
wsl -d Ubuntu
source ~/spacenet-env/bin/activate
cd ~/p2
```

o LD_LIBRARY_PATH pro CUDA ja ta configurado no activate do venv.
