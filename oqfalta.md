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
- [x] treino resize completo e comparado com tiling (F1 0.56 vs 0.62)
- [x] dataset_check.ipynb (estrutura, tif preto, tif->geojson->mascara, distribuicao)
- [x] relatorio.ipynb (codigo + resultados + discussao, metricas ao vivo)

## pendente

### entregas da disciplina
- [ ] video explicativo (< 10 min)
- [ ] resumo 1 pagina 2 colunas (pdf)

### revisao final do relatorio
- [ ] validar a escrita do markdown do relatorio (revisar texto, sem erro)
- [ ] confirmar que o codigo do notebook esta validado (roda sem erro)
- [ ] procurar referencias dos slides da disciplina e do que a gente usou
      (U-Net, ResNet, SpaceNet, livro do Gonzalez) e citar direito

### entrega
- [ ] arrumar o repositorio github de entrega do projeto

### melhorias possiveis (nao obrigatorias)
- [ ] threshold otsu no pos-processamento (em vez de 0.5 fixo)
- [ ] ajustar min_area e morph_kernel pro dataset real
- [ ] visualizacao no QGIS (abrir predictions.geojson)
- [ ] treino com dados do test oficial (381 tiles sem label, fine-tuning)
