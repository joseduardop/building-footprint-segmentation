# o que falta

## feito

- [x] pipeline completa rodando (treino, postprocess, evaluate)
- [x] treino com GPU (RTX 3060 Ti, 37 epocas, val IoU 0.7251)
- [x] resultados: F1 0.62, precision 0.70, recall 0.55, IoU 0.72
- [x] predictions.geojson gerado pro test split (173 tiles, 1759 predios)
- [x] figuras: overlays, curvas de treino, grid do dataset
- [x] demo.ipynb funcionando (tiles + google maps)
- [x] cache de mascaras (create_masks.py)
- [x] previews png dos tiles 16-bit
- [x] modo tiling com overlap 50%
- [x] modo resize implementado (nao treinado ainda)
- [x] validacao no tile inteiro (nao so o centro)
- [x] filtro por split no postprocess e evaluate
- [x] traduzido pra portugues

## pendente

### entregas da disciplina
- [ ] video explicativo (< 10 min)
- [ ] resumo 1 pagina 2 colunas (pdf)
- [ ] relatorio.ipynb (notebook com codigo + resultados + discussao)

### experimento resize
- [ ] rodar treino com `--mode resize` e comparar com tiling
- [ ] documentar a diferenca no relatorio (tiling preserva detalhe
      mas tem emendas; resize perde detalhe mas e mais simples)

### notebook de validacao do dataset
- [ ] dataset_check.ipynb: estrutura de pastas, por que o tif abre preto,
      relacao tif -> geojson -> mascara, distribuicao de predios por tile

### melhorias possiveis (nao obrigatorias)
- [ ] threshold otsu no pos-processamento (em vez de 0.5 fixo)
- [ ] ajustar min_area e morph_kernel pro dataset real
- [ ] visualizacao no QGIS (abrir predictions.geojson)
- [ ] treino com dados do test oficial (381 tiles sem label, fine-tuning)
