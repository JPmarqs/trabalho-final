# Regressão do valor FOB (log1p(VL_FOB)) com MLP — Keras

**Disciplina:** Deep Learning — Trabalho final
**Frente escolhida:** 3.1 Multilayer Perceptrons
**Fonte dos dados:** Secex/MDIC — Exportação por Município (SH4), 2015–2025

---

## 1. Formulação do problema

### 1.1 Contexto e utilidade

Toda declaração de exportação no Brasil informa o valor da operação em
US$ FOB. Estimar esse valor a partir das características da operação —
produto, destino, origem, calendário e volume — é útil para **auditoria
alfandegária em lote** (a Receita compara o valor declarado com o valor
esperado para o perfil; desvios sinalizam sub/sobre-faturamento) e para
**estimativa prévia de receita** por parte de exportadores e órgãos de
fomento antes do embarque.

### 1.2 Unidade de observação e variável-alvo

- **Unidade de observação:** uma linha da Secex — agregado
  município × SH4 × país × mês.
- **Variável-alvo:** `y = log1p(VL_FOB)` — valor da exportação em US$,
  transformado por log1p (a distribuição original tem assimetria ≈ 165 e
  valores até US$ 1,9 bi; o log mede erro *proporcional* e evita que a
  MAE seja dominada pelas transações bilionárias).
- **Tipo do problema:** **regressão** (alvo contínuo).

### 1.3 Atributos de entrada e ausência de vazamento

| Atributo | Tipo | Transformação | Justificativa |
|---|---|---|---|
| `SH4` (~1200 níveis) | categórica | one-hot | identifica a mercadoria — preço varia fortemente por produto; é conhecida *antes* de precificar/declarar |
| `CO_PAIS` | categórica | one-hot | destino; incidência tarifária e frete variam por mercado |
| `SG_UF_MUN` | categórica | one-hot | origem; custo logístico e produto regional |
| `CO_MES` | cíclica | seno/cosseno | estacionalidade de preços e volumes |
| `CO_ANO` | numérica | padronização | inflação/câmbio ao longo do tempo |
| `KG_LIQUIDO` | numérica | `log1p` + padronização + indicador | o principal determinante do valor; indicador marca peso 0/ausente (7,2% das linhas) |

**Diferença crucial em relação ao problema anterior:** aqui `VL_FOB` é o
**alvo** e `SH4` é entrada **legítima** — o cenário de uso (estimar o
valor de uma mercadoria já identificada) é realista. Nada derivado de
`VL_FOB` entra entre as entradas; não há vazamento.

## 2. Dados e análise exploratória

A EDA (`relatorios/eda_out/`, gerada por `src/analise.py`) sustenta as
decisões: `VL_FOB` tem assimetria ≈ 165 e curtose ≈ 40.523 → transformação
logarítmica; `KG_LIQUIDO` assimetria ≈ 130 → `log1p` + padronização;
7,2% dos pesos são zero/ausentes → indicador binário; a sazonalidade
mensal (pico em julho/outubro) motiva a codificação cíclica do mês.

Estatísticas do alvo por partição (`distribuicao_alvo.csv`):

| Partição | n | média (log) | desvio (log) | mediana (log) | p90 (log) | máx (log) |
|---|---|---|---|---|---|---|
| Treino 2015–23 | 450.000 | 8,09 | 3,19 | 8,12 | 12,16 | 20,65 |
| Validação 2024 | 50.000 | 7,84 | 3,26 | 7,67 | 12,13 | 20,72 |
| Teste 2025 | 50.000 | 7,84 | 3,24 | 7,68 | 12,13 | 19,98 |

A média do treino é ~0,25 nat-log acima da validação/teste — deriva
temporal real (a média do preditor treinado é enviesada em 2024/25), que
o modelo precisa compensar com as demais features.

## 3. Protocolo experimental

| Partição | Anos | Finalidade |
|---|---|---|
| Treinamento | 2015–2023 | ajustar pesos e estimar transformações |
| Validação | 2024 | busca de hiperparâmetros e parada antecipada |
| Teste | 2025 | desempenho final (uma única avaliação) |

- Divisão **cronológica**, amostra de até 50.000 linhas/ano (semente 42).
- Todas as transformações (categorias, média/desvio de ano, peso **e do
  alvo**) ajustadas **somente no treino**.
- **Perda:** MSE sobre o alvo padronizado. **Métrica da busca:** MAE de
  validação em unidades log1p (minimizar).
- **Métricas de avaliação:** MAE, RMSE e R² na escala log1p (unidade do
  alvo transformado) + erro percentual mediano e P90 em US$ (interpretação
  na unidade original).
- **Modelo:** MLP Keras (`Dense+ReLU` por camada oculta, `Dropout`, L2,
  saída linear, Adam), **parada antecipada** (paciência 5, MAE de
  validação, restauração dos melhores pesos).
- **Busca Optuna** (TPE + poda mediana, 12 tentativas): lr
  $[10^{-4},3\cdot10^{-3}]$, 1–3 camadas, largura {64,128,256,512},
  dropout 0–0,4, L2 $[10^{-6},10^{-3}]$, batch {256,512,1024}.
- **Comparação:** top-3 retreinadas + baseline sem regularização
  (dropout=L2=0) + dois baselines ingênuos (preditor médio; reta
  fechada `log1p(VL_FOB) ~ log1p(KG_LIQUIDO)`).

Implementação: `src/treinar_regressao_keras.py` (Keras 3.16 /
TensorFlow 2.22). Saídas em `relatorios/regressao_keras/` (MSE) e
`relatorios/regressao_keras_huber/` (Huber).

## 4. Resultados

### 4.0 Comparação de perdas: Huber vs MSE

Como `VL_FOB` tem cauda longa (assimetria 120,6 bruta; ver EDA), a perda
**Huber** foi implementada à mão em `loss_huber()` — quadrática dentro de
δ, linear fora — e `delta` foi incluído no espaço de busca do Optuna
($[0{,}1; 3]$ log, em unidades de desvio-padrão do alvo padronizado;
$1σ ≈ 3{,}2$ no log1p). A implementação foi validada contra a fórmula
analítica e contra `keras.losses.Huber` (perda e gradiente
$\mathrm{sign}(e)\cdot\min(|e|,\delta)$ coincidem). A versão MSE foi
preservada em `relatorios/regressao_keras/` (`--loss mse`).

| | **MSE** (`regressao_keras/`) | **Huber** (`regressao_keras_huber/`) |
|---|---|---|
| Melhor MAE validação (busca) | 0,7464 | 0,7455 |
| **MAE teste** | **0,7760** | 0,7930 |
| **RMSE teste** | **1,121** | 1,145 |
| **R² teste** | **0,881** | 0,875 |
| Erro mediano (US$) | **51,3%** | 54,7% |
| Erro P90 (US$) | **275%** | 324% |
| δ escolhido nos top-3 | — | 0,12 / 0,43 / 1,56 |

**O Huber não melhorou os resultados** — a validação empatou (0,7455 vs
0,7464) e o teste piorou levemente em todas as métricas. Duas leituras:

1. **A cauda já foi tratada pelo `log1p`.** O Huber protege o treino de
   erros grandes na escala *bruta*, mas aqui o alvo é logarítmico: o
   scatter predito×real mostra dispersão praticamente homocedástica ao
   longo da diagonal (log 0–20), ou seja, não sobrou cauda pesada na
   escala em que a perda atua. O Huber teve pouco a fazer.
2. **A busca puxou δ para baixo** (0,12 e 0,43 nos dois melhores
   trials — bem abaixo de 1σ). δ pequeno ≈ perda MAE: o Optuna estava
   *otimizando* exatamente a métrica de avaliação na validação, mas essa
   robustez adicional não transferiu para o teste — os resíduos grandes
   que o Huber ignora são justamente os de transações grandes, cujo
   acerto contribui para MAE/RMSE no teste.

Conclusão acadêmica: **a transformação `log1p` do alvo já cumpriu o
papel que o Huber cumpre em problemas com cauda bruta**; as duas
técnicas são substitutas aqui, não complementares, e o `log1p` +
MSE venceu por pouco. O experimento vale como evidência empírica — não
como regressão.

### 4.1 Busca de hiperparâmetros (MSE, referência)

12 tentativas, todas concluídas (`resultados_busca.csv`); MAE de
validação entre 0,750 e 0,83. As 4 melhores convergiram para lr
≈ 2,7–3,3·10⁻⁴, largura 128 (ou 512), batch 256, dropout 0,15–0,24 e L2
muito pequeno (10⁻⁶–10⁻⁵). A profundidade 2–3 venceu a rasa (1 camada).

### 4.2 Comparação no teste (2025)

| Modelo | Camadas | Parâmetros | MAE val (log) | **MAE teste** | **RMSE teste** | **R² teste** | Erro mediano US$ |
|---|---|---|---|---|---|---|---|
| **top2** (vencedor) | 2×128 | 204.801 | **0,746** | **0,776** | **1,121** | **0,881** | **51,3%** |
| top11 | 3×128 | 221.313 | 0,761 | 0,789 | 1,133 | 0,878 | 51,9% |
| top10 | 3×128 | 221.313 | 0,762 | 0,791 | 1,136 | 0,877 | 52,2% |
| baseline sem reg. | 2×128 | 204.801 | 0,785 | 0,827 | 1,161 | 0,872 | 60,8% |
| *reta só-peso* | — | 2 | — | 1,301 | — | 0,735 | — |
| *preditor médio* | — | 1 | — | 2,711 | — | −0,006 | — |

### 4.3 Curvas e diagnóstico

Figura `curvas_treinamento.png`:

- A perda de treino segue caindo enquanto a perda/MAE de validação estabiliza
  por volta da época 5–10 — **overfitting leve**, contido pela parada
  antecipada (épocas 9–28 conforme o modelo).
- O scatter predito×real mostra aderência forte ao longo de toda a
  faixa (0 a 20 em log), com dispersão crescente nas caudas — as
  predições de transações muito grandes são as menos precisas
  (P90 do erro = 275%).
- O baseline sem dropout/L2 atinge o pior MAE entre os quatro e para
  na época 9 — aqui **a regularização ajudou de fato** (diferente do
  problema de classificação anterior): 0,776 vs 0,827 de MAE.

## 5. Interpretação dos erros na unidade do alvo

- **MAE 0,776 em log1p** significa erro médio multiplicativo: $e^{0{,}776}
  ≈ 2{,}2$ — em média, a previsão erra por um fator ~2,2× o valor real.
  Para um produto com erro na mediana (51%), a previsão declararia
  US$ 150 mil em vez de US$ 100 mil.
- **R² = 0,88** na escala log: 88% da variância do valor logaritmizado
  é explicada pelas features — contra 74% da reta que usa *apenas* o peso
  e ~0% do preditor médio. Os atributos categóricos (produto, destino,
  origem) somam ~14 pontos de R² sobre o peso sozinho.
- **RMSE 1,12 > MAE 0,78** indica caudas: os erros grandes (transações
  bilionárias) puxam o quadrado médio — é esperado e é o motivo de
  reportar também o erro mediano.

## 6. Limitações

1. Unidade agregada: linhas com atributos idênticos têm valores
   diferentes (ruído irreducível).
2. Erro percentual alto em transações extremas; para auditoria, usaria-se
   o modelo como *triagem* (desvio > limiar → revisão), não como veredito.
3. Amostragem de 50k/ano e um único ano de teste limitam a precisão da
   estimativa; a deriva temporal (média do treino ≠ média de 2024/25)
   sugere re-treino periódico.
4. `CO_MUN` (alta cardinalidade) ficou de fora; embeddings de município
   e de SH4 poderiam reduzir a dimensão one-hot (~1250 features).

## 7. Conclusão

Sob o mesmo protocolo cronológico do trabalho, a MLP Keras atingiu
**R² = 0,88, MAE = 0,78 (log1p) e erro mediano de 51% em US$** no teste
de 2025, superando a reta de peso sozinho (R² = 0,74) e o preditor médio
(R² ≈ 0) — ou seja, produto, destino, origem e calendário agregam
informação real sobre o valor. A comparação de arquiteturas mostrou
retorno decrescente (2×128 ≥ 3×128 ≥ baseline), overfitting leve
controlado por parada antecipada, e ganho claro da regularização
(dropout/L2) sobre a versão sem regularização — ao contrário do que se
observou no problema de classificação anterior.

## Anexos

- `regressao_keras/` — resultados com perda MSE.
- `regressao_keras_huber/` — resultados com perda Huber (δ na busca).
- `regressao_keras/resultados_busca.csv` — tentativas do Optuna.
- `regressao_keras/metricas_teste.json` — métricas por modelo + baselines.
- `regressao_keras/historico_treino.csv` — curvas época a época.
- `regressao_keras/predicoes_teste.csv` — predições linha a linha.
- `regressao_keras/distribuicao_alvo.csv` — estatísticas do alvo.
- `regressao_keras/curvas_treinamento.png` — figura consolidada.
- `regressao_keras/log_execucao.txt` — log completo da execução.
