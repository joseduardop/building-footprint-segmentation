# Roteiro do Vídeo - Segmentação de Edificações (SpaceNet 2, Paris)

EEL7815 - Processamento Digital de Imagens — UFSC, 2026.1
Duração alvo: **menos de 10 minutos**

Estrutura abaixo: **[tempo]** | **o que mostrar na tela** | **o que falar**.
A fala está em tom de apresentação, pode adaptar pras suas palavras.

---

## 1. Contexto e problemática — [~1 min]

**Tela:** uma imagem de satélite de um bairro (pode ser um tile do dataset ou
uma captura de satélite externa).

**Fala:**
"Oi, eu sou o José Eduardo, e esse é o meu projeto da disciplina de Processamento
Digital de Imagens. A ideia parte de um problema bem prático: dá pra olhar uma
imagem de satélite de uma cidade e identificar automaticamente onde estão as
edificações? Isso é útil pra cartografia, planejamento urbano e até resposta a
desastres — situações onde mapear construções na mão seria inviável. O desafio é
que isso é uma tarefa de segmentação: não basta dizer 'tem prédio na imagem', a
gente precisa marcar exatamente quais pixels são edificação e quais são fundo."

---

## 2. Proposta e objetivo — [~1 min]

**Tela:** um slide simples com o fluxo: imagem de satélite -> modelo -> polígonos
no mapa. (Pode usar o diagrama do README.)

**Fala:**
"A proposta é montar uma pipeline completa: a entrada é a imagem de satélite e a
saída são polígonos georreferenciados, ou seja, contornos de cada prédio já em
latitude e longitude, prontos pra abrir num programa de mapas tipo o QGIS. Pra
isso combino técnicas clássicas de PDI — realce de contraste, limiarização e
morfologia — com uma rede neural convolucional que faz a segmentação. O objetivo
não é só treinar um modelo, é ter o caminho inteiro funcionando, da imagem crua
até o resultado vetorial."

---

## 3. Dataset — [~1 min 30]

**Tela:** abrir o `notebooks/dataset_check.ipynb` e mostrar o grid de tiles
(`figures/dataset_grid.png`). Mostrar também um tif "preto" e o mesmo depois de
normalizado.

**Fala:**
"O dataset é o SpaceNet 2, recorte de Paris. São 1148 imagens de 650 por 650
pixels, com 16 bits e três bandas de cor, e cada uma vem com um arquivo GeoJSON
que marca as edificações. Um detalhe importante de PDI aqui: essas imagens são de
16 bits, então se a gente abre direto elas aparecem quase pretas, porque os
valores ficam concentrados numa faixa estreita. Por isso aplico uma normalização
por percentil, que é um realce radiométrico — ele espalha o contraste e a imagem
fica visível. Divido os dados em 70% pra treino, 15% pra validação e 15% pra
teste."

---

## 4. Organização de arquivos — [~1 min]

**Tela:** mostrar a árvore de pastas do projeto (README seção de organização):
`scripts/`, `util/`, `notebooks/`, `artefatos/`, `figures/`.

**Fala:**
"O projeto é organizado em scripts independentes, cada um acionável por linha de
comando. Em `scripts` ficam as quatro etapas da pipeline: `dataset` carrega e
prepara os dados, `train` treina o modelo, `postprocess` faz a inferência e gera
os polígonos, e `evaluate` calcula as métricas. Em `util` ficam funções de apoio,
como a geração das máscaras e das figuras. Os notebooks são só pra análise e demo,
o código pesado tá todo em `.py`. Os modelos e resultados gerados ficam em
`artefatos`, e as figuras em `figures`."

---

## 5. Arquitetura proposta — [~1 min 30]

**Tela:** diagrama da U-Net (README seção arquitetura) e o trecho do código do
`build_model`. Mostrar também os dois modos: tiling e resize.

**Fala:**
"O modelo é uma U-Net, uma arquitetura clássica de segmentação em formato de U:
um codificador que vai comprimindo a imagem e extraindo características, e um
decodificador que reconstrói a máscara na resolução original, com conexões de
salto que preservam o detalhe. Como codificador uso uma ResNet34 já pré-treinada
no ImageNet — isso é transfer learning, aproveito o que ela já aprendeu sobre
imagens em geral. Treino com uma perda combinada de entropia cruzada e Dice, que
é boa pra segmentação. Comparei duas formas de tratar a resolução: o 'tiling',
que recorta a imagem em pedaços de 256 mantendo o detalhe original, e o 'resize',
que encolhe a imagem inteira. Depois da rede, o pós-processamento aplica
limiarização e morfologia pra limpar a máscara e extrair os contornos."

---

## 6. Resultados — [~1 min 30]

**Tela:** abrir o `relatorio.ipynb` na seção de resultados — mostrar as curvas de
treino e a tabela de métricas. Mostrar uma figura de comparação
(`figures/compare_imgXXX.png`) com original, resize, tiling e gabarito.

**Fala:**
"Nos resultados, o tiling ganhou em todas as métricas: F1 de 0,62 contra 0,56 do
resize, e IoU médio de 0,72. A diferença maior foi na revocação — ou seja, o
tiling encontra mais prédios. Faz sentido: quando a gente encolhe a imagem no
resize, as construções pequenas somem e o modelo não consegue mais detectar. Aqui
nessa comparação visual dá pra ver lado a lado a imagem original, a detecção do
resize em vermelho, a do tiling em azul, e o gabarito em verde. O tiling
acompanha bem melhor os prédios menores."

---

## 7. Demo — [~1 min 30]

**Tela:** rodar o `notebooks/demo.ipynb` ao vivo: carregar o modelo, escolher uma
imagem (um tile de teste e, se der, uma imagem de satélite externa), rodar a
inferência e mostrar o resultado sobreposto.

**Fala:**
"Pra fechar, uma demonstração ao vivo. Aqui eu carrego o modelo treinado e passo
uma imagem que ele nunca viu. Ele roda a inferência e devolve a imagem com as
edificações marcadas. Testei também com imagens de satélite de fora do dataset,
de outras regiões, e ele generaliza bem — desde que o nível de zoom seja parecido
com o do treino, porque o modelo é sensível à escala. Isso, inclusive, é um dos
pontos que listei como trabalho futuro: treinar com múltiplas escalas pra deixar
ele mais robusto. E é isso, obrigado!"

---

## Dicas de gravação

- Roteiro soma ~9 min — tem folga pros 10. Se precisar cortar, encurte a parte de
  organização de arquivos (item 4).
- Grave a demo (item 7) **antes**, já com o modelo carregado, pra não perder tempo
  esperando o carregamento no vídeo.
- Se a internet/gravação travar na demo ao vivo, deixe um resultado já gerado de
  backup pra mostrar.
- Fale olhando o resultado na tela, não leia o roteiro palavra por palavra.
