# Regressão do valor FOB com MLP (Keras)

Este projeto formula um problema de **regressão** sobre os dados públicos da
Secex/MDIC (Exportação por Município, 2015–2025): **prever `log1p(VL_FOB)`**
(o valor da exportação em US$, logaritmizado) a partir de informações
conhecidas antes de fechar/declarar o negócio:

| Atributo | Tipo | Transformação |
|---|---|---|
| `SH4` (produto) | categórica (~1200 níveis) | one-hot |
| `CO_PAIS` (destino) | categórica | one-hot |
| `SG_UF_MUN` (origem) | categórica | one-hot |
| `CO_MES` | cíclica | seno/cosseno |
| `CO_ANO` | numérica | padronização |
| `KG_LIQUIDO` | numérica | `log1p` + padronização + indicador de peso ausente/zero |

Justificativa: ao contrário do problema anterior (prever SH2), aqui
`VL_FOB` é o **alvo** e `SH4` é entrada legítima — o cenário de uso é
auditoria/estimativa de valor de uma mercadoria já identificada. Não há
vazamento: nada derivado de `VL_FOB` entra entre as entradas.

## Protocolo experimental

- **Treino:** 2015–2023; **validação:** 2024; **teste final:** 2025.
- A divisão é cronológica, sem observações futuras no treino.
- One-hot de SH4/país/UF, seno/cosseno do mês, ano padronizado e
  `KG_LIQUIDO` por `log1p` + padronização + indicador de peso ruim.
- Categorias, média e desvio-padrão (inclusive do alvo) são ajustados
  somente no treino.
- Perda: **Huber** (implementada à mão em `loss_huber()` — quadrática
  dentro de δ, linear fora; δ é hiperparâmetro da busca) sobre o alvo
  padronizado; `--loss mse` reproduce a versão anterior. Motivo: `VL_FOB`
  tem cauda longa (assimetria 120,6 bruta; transações de bilhões) e a
  região linear do Huber ignora os erros grandes dessas caudas.
  Métricas de avaliação: **MAE, RMSE e R² na escala log1p**, mais erro
  percentual mediano/P90 em US$ (interpretados na unidade da
  variável-alvo).
- Busca de hiperparâmetros com **Optuna** (TPE + poda mediana),
  minimizando o MAE de validação; espaço: taxa de aprendizado, nº de
  camadas, largura, dropout, L2, batch size e **δ do Huber**.
- As 3 melhores configurações são retreinadas com parada antecipada
  (MAE de validação) e comparadas no teste, além de um baseline sem
  regularização e dois baselines ingênuos (preditor médio e reta
  `log1p(VL_FOB) ~ log1p(KG_LIQUIDO)` ajustada no treino).
- Modelo: MLP Keras (Dense+ReLU+Dropout+L2, saída linear), Adam.

## Execução

O comando abaixo executa o experimento com até 50 mil linhas por ano,
12 tentativas de Optuna e até 30 épocas:

```powershell
& ".\.venv\Scripts\python.exe" ".\src\treinar_regressao_keras.py"
```

Para usar todas as linhas e mais tentativas:

```powershell
& ".\.venv\Scripts\python.exe" ".\src\treinar_regressao_keras.py" `
  --max-por-ano 0 --tentativas 20 --epocas 40
```

Para reproduzir a versão com MSE (sem Huber):

```powershell
& ".\.venv\Scripts\python.exe" ".\src\treinar_regressao_keras.py" `
  --loss mse --out ".\relatorios\regressao_keras"
```

Os resultados ficam em `relatorios/regressao_keras_huber/` (Huber, padrão)
e `relatorios/regressao_keras/` (MSE): `resultados_busca.csv`,
`metricas_teste.json`, `historico_treino.csv`, `particoes.csv`,
`distribuicao_alvo.csv`, `predicoes_teste.csv`, `curvas_treinamento.png`
e `log_execucao.txt`.

O relatório EDA em `relatorios/eda_out/` (gerado por `src/analise.py`)
documenta a qualidade dos dados, distribuições, valores ausentes,
anomalias, sazonalidade e concentração de capítulos por UF — incluindo a
análise de `VL_FOB` (assimetria ≈ 165, outliers de preço/kg) que
fundamenta a transformação logarítmica do alvo.
