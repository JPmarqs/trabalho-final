#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
  MLP com Keras — REGRESSÃO de log1p(VL_FOB) (Secex/MDIC)
  Disciplina: Deep Learning — Frente: Multilayer Perceptrons
=====================================================================
Problema:
  Prever o valor FOB (US$) de uma operação de exportação a partir de
  informações conhecidas antes de fechar/declarar o negócio:
  produto (SH4), país de destino, UF do município embarcador, mês, ano
  e volume (KG_LIQUIDO).  Alvo = log1p(VL_FOB)  → regressão.

  Justificativa de uso das colunas: ao contrário do problema anterior
  (prever SH2), aqui VL_FOB é o ALVO e SH4 é entrada legítima — o
  cenário de uso é auditoria/estimativa de valor de mercadoria já
  identificada.  Não há vazamento: nada derivado de VL_FOB entra nas
  entradas.

Protocolo (mesmo que a versão de classificação):
  - Treino 2015–2023 | Validação 2024 | Teste 2025 (cronológico).
  - Até 50.000 linhas por ano, semente 42.
  - One-hot de SH4/CO_PAIS/SG_UF_MUN, seno/cosseno do mês, ano
    padronizado, log1p(KG_LIQUIDO) padronizado + indicador de peso
    ausente/zero.  Estatísticas ajustadas SOMENTE no treino.
  - Alvo padronizado com média/desvio do treino; perda HubER (escrita à
    mão, delta buscado pelo Optuna) ou MSE, via --loss.
  - Busca Optuna (TPE + poda) minimizando MAE de validação (escala
    log1p); parada antecipada com restauração dos melhores pesos.
  - Top-3 retreinadas + baseline sem regularização + baselines
    ingênuos (média e regressão linear só com o peso).

Uso:
  python treinar_regressao_keras.py                     # 50k/ano, 12 trials
  python treinar_regressao_keras.py --max-por-ano 0     # todos os dados
Saídas (em --out):
  resultados_busca.csv, metricas_teste.json, historico_treino.csv,
  particoes.csv, distribuicao_alvo.csv, predicoes_teste.csv,
  curvas_treinamento.png, optuna.db, log_execucao.txt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.get_logger().setLevel("ERROR")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ----------------------------------------------------------------------
# 0. CONFIGURAÇÃO
# ----------------------------------------------------------------------
COLS = ["CO_ANO", "CO_MES", "SH4", "CO_PAIS", "SG_UF_MUN",
        "KG_LIQUIDO", "VL_FOB"]
ANOS_TREINO = list(range(2015, 2024))
ANO_VAL = 2024
ANO_TESTE = 2025
N_EPOCAS_MAX = 30
PACIENCIA = 5


# ----------------------------------------------------------------------
# 1. DADOS
# ----------------------------------------------------------------------
def carregar_dados(dir_dados: Path, max_por_ano: int, semente: int) -> pd.DataFrame:
    partes = []
    for ano in ANOS_TREINO + [ANO_VAL, ANO_TESTE]:
        csv = dir_dados / f"EXP_{ano}_MUN.csv"
        if not csv.exists():
            sys.exit(f"[erro] arquivo não encontrado: {csv}")
        df = pd.read_csv(csv, sep=";", dtype=str, usecols=COLS)
        for c in ["CO_ANO", "CO_MES"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
        for c in ["KG_LIQUIDO", "VL_FOB"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["CO_PAIS"] = df["CO_PAIS"].str.zfill(3)
        df["SH4"] = df["SH4"].str.zfill(4)
        df = df.dropna(subset=["VL_FOB"])
        df = df[df.VL_FOB >= 0]                    # sem valores negativos
        if max_por_ano and len(df) > max_por_ano:
            df = df.sample(max_por_ano, random_state=semente)
        df["particao"] = np.where(
            df.CO_ANO <= ANOS_TREINO[-1], "treino",
            np.where(df.CO_ANO == ANO_VAL, "validacao", "teste"))
        partes.append(df)
        print(f"  {ano}: {len(df):,} linhas ({df.particao.iloc[0]})")
    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO (ajustado APENAS no treino)
# ----------------------------------------------------------------------
class Preprocessador:
    """One-hot de SH4/país/UF + seno-cosseno do mês + ano e log(peso)
    padronizados + indicador de peso ruim. Alvo padronizado separadamente."""

    def __init__(self):
        self.sh4: list[str] = []
        self.paises: list[str] = []
        self.ufs: list[str] = []
        self.ano_mu = self.ano_sd = None
        self.logkg_mu = self.logkg_sd = None
        self.y_mu = self.y_sd = None

    @staticmethod
    def _onehot(valores: pd.Series, categorias: list[str]) -> np.ndarray:
        cat = pd.Categorical(valores, categories=categorias)
        X = np.zeros((len(cat), len(categorias)), dtype=np.float32)
        ok = cat.codes >= 0
        X[np.nonzero(ok)[0], cat.codes[ok]] = 1.0
        return X

    def fit(self, treino: pd.DataFrame):
        self.sh4 = sorted(treino.SH4.dropna().unique())
        self.paises = sorted(treino.CO_PAIS.dropna().unique())
        self.ufs = sorted(treino.SG_UF_MUN.dropna().unique())
        self.ano_mu, self.ano_sd = treino.CO_ANO.mean(), treino.CO_ANO.std()
        logkg = np.log1p(treino.KG_LIQUIDO.clip(lower=0).fillna(0.0))
        self.logkg_mu, self.logkg_sd = logkg.mean(), logkg.std()
        y = np.log1p(treino.VL_FOB.to_numpy(np.float64))
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())
        return self

    def transform_X(self, df: pd.DataFrame) -> np.ndarray:
        X_sh4 = self._onehot(df.SH4.fillna(""), self.sh4)
        X_pais = self._onehot(df.CO_PAIS.fillna(""), self.paises)
        X_uf = self._onehot(df.SG_UF_MUN.fillna(""), self.ufs)
        ang = 2 * np.pi * df.CO_MES.to_numpy() / 12.0
        X_mes = np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)
        X_ano = ((df.CO_ANO - self.ano_mu) / self.ano_sd).to_numpy(
            np.float32).reshape(-1, 1)
        kg = df.KG_LIQUIDO.clip(lower=0).fillna(0.0)
        X_logkg = ((np.log1p(kg) - self.logkg_mu) / self.logkg_sd).to_numpy(
            np.float32).reshape(-1, 1)
        X_ruim = (df.KG_LIQUIDO.isna() | (df.KG_LIQUIDO <= 0)).to_numpy(
            np.float32).reshape(-1, 1)
        return np.hstack([X_sh4, X_pais, X_uf, X_mes, X_ano, X_logkg, X_ruim])

    def transform_y(self, df: pd.DataFrame) -> np.ndarray:
        y = np.log1p(df.VL_FOB.to_numpy(np.float64))
        return ((y - self.y_mu) / self.y_sd).astype(np.float32)

    def despadronizar(self, y_std: np.ndarray) -> np.ndarray:
        """De volta para a escala log1p(VL_FOB)."""
        return y_std * self.y_sd + self.y_mu


# ----------------------------------------------------------------------
# 3. MÉTRICAS DE REGRESSÃO (escala log1p, a menos que indicado)
# ----------------------------------------------------------------------
def metricas_reg(y_real: np.ndarray, y_pred: np.ndarray,
                 y_usd_real: np.ndarray | None = None,
                 y_usd_pred: np.ndarray | None = None) -> dict:
    erro = y_pred - y_real
    mae = float(np.mean(np.abs(erro)))
    rmse = float(np.sqrt(np.mean(erro ** 2)))
    r2 = float(1.0 - np.sum(erro ** 2) /
               max(np.sum((y_real - y_real.mean()) ** 2), 1e-12))
    m = {"mae_log": mae, "rmse_log": rmse, "r2_log": r2}
    if y_usd_real is not None and y_usd_pred is not None:
        v = y_usd_real > 0
        ape = np.abs(y_usd_pred[v] - y_usd_real[v]) / y_usd_real[v]
        m["erro_pct_mediano"] = float(np.median(ape) * 100)
        m["erro_pct_p90"] = float(np.quantile(ape, 0.90) * 100)
    return m


# ----------------------------------------------------------------------
# 4. MODELO KERAS + CALLBACK DE VALIDAÇÃO
# ----------------------------------------------------------------------
def loss_huber(delta: float):
    """Loss Huber implementada à mão (roda na autograd do TF):

        L(e) = 0,5·e²              se |e| ≤ delta        (quadrática)
        L(e) = delta·(|e| − 0,5·delta)  caso contrário   (linear)

    com e = y_true − y_pred no alvo padronizado. A região linear torna a
    perda robusta aos erros grandes vindos da cauda longa de VL_FOB
    (petróleo/minérios ≥ US$ 100 mi), sem perder a precisão quadrática
    nas remessas comuns. `delta` é buscado pelo Optuna.

    Equivale a keras.losses.Huber(delta), mas escrita explicitamente para
    deixar a matemática visível no código do trabalho.
    """
    d = tf.constant(delta, dtype=tf.float32)

    def huber(y_true, y_pred):
        erro = tf.abs(y_true - y_pred)
        quadratica = 0.5 * tf.square(erro)
        linear = d * (erro - 0.5 * d)
        return tf.where(erro <= d, quadratica, linear)

    huber.__name__ = "huber_custom"
    return huber


def construir_modelo(n_feat: int, n_ocultas: int, largura: int,
                     dropout: float, l2: float, lr: float,
                     loss_tipo: str = "huber",
                     huber_delta: float = 1.0) -> keras.Model:
    entradas = keras.Input(shape=(n_feat,), name="entrada")
    x = entradas
    reg = regularizers.l2(l2) if l2 > 0 else None
    for i in range(n_ocultas):
        x = layers.Dense(largura, activation="relu", kernel_regularizer=reg,
                         name=f"densa_{i+1}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)
    saida = layers.Dense(1, name="saida")(x)       # saída linear p/ regressão
    modelo = keras.Model(entradas, saida, name="mlp_regressao")
    loss = (loss_huber(huber_delta) if loss_tipo == "huber" else "mse")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                   loss=loss, metrics=["mae"])
    return modelo


class CallbackoValidacao(keras.callbacks.Callback):
    """MAE de validação em unidades log1p(VL_FOB) por época; parada
    antecipada com restauração dos melhores pesos; poda Optuna opcional."""

    def __init__(self, X_va, y_va, y_sd: float, pre: Preprocessador,
                 paciencia: int, nome: str = "",
                 trial: "optuna.Trial | None" = None):
        super().__init__()
        self.X_va, self.y_va, self.y_sd, self.pre = X_va, y_va, y_sd, pre
        self.paciencia, self.nome, self.trial = paciencia, nome, trial
        self.melhor_mae, self.sem_melhora = np.inf, 0
        self.melhores_pesos = None
        self.historico: list[dict] = []
        self.podado = False

    def on_epoch_end(self, ep, logs=None):
        t0 = time.perf_counter()
        pred = self.model(self.X_va, training=False).numpy().ravel()
        mae_log = float(np.mean(np.abs(self.pre.despadronizar(pred)
                                       - self.pre.despadronizar(self.y_va))))
        reg = {"modelo": self.nome, "epoca": ep + 1,
               "perda_treino": float(logs["loss"]),
               "perda_validacao": float(logs["val_loss"]),
               "mae_validacao": mae_log,
               "segundos": time.perf_counter() - t0}
        self.historico.append(reg)
        print(f"    {self.nome} ep{ep+1:02d}: perda_t={reg['perda_treino']:.4f} "
              f"perda_v={reg['perda_validacao']:.4f} mae_v={mae_log:.4f} "
              f"({reg['segundos']:.1f}s)")
        if self.trial is not None:
            self.trial.report(mae_log, ep + 1)
            if self.trial.should_prune():
                self.podado = True
                self.model.stop_training = True
                return
        if mae_log < self.melhor_mae - 1e-4:
            self.melhor_mae, self.sem_melhora = mae_log, 0
            self.melhores_pesos = self.model.get_weights()
        else:
            self.sem_melhora += 1
            if self.sem_melhora >= self.paciencia:
                print(f"    {self.nome}: parada antecipada na época {ep+1} "
                      f"(melhor MAE validação = {self.melhor_mae:.4f})")
                self.model.stop_training = True


def treinar_modelo(modelo, X_tr, y_tr, X_va, y_va, pre: Preprocessador,
                   nome: str, semente: int, batch_size: int = 512):
    tf.keras.utils.set_random_seed(semente)
    cb = CallbackoValidacao(X_va, y_va, pre.y_sd, pre, PACIENCIA, nome)
    modelo.fit(X_tr, y_tr, validation_data=(X_va, y_va),
               epochs=N_EPOCAS_MAX, batch_size=batch_size,
               callbacks=[cb], verbose=0)
    if cb.melhores_pesos is not None:
        modelo.set_weights(cb.melhores_pesos)
    return cb


# ----------------------------------------------------------------------
# 5. EXPERIMENTO
# ----------------------------------------------------------------------
def main():
    raiz = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default=str(raiz / "datasets"))
    ap.add_argument("--out", default=str(raiz / "relatorios" / "regressao_keras"))
    ap.add_argument("--max-por-ano", type=int, default=50_000)
    ap.add_argument("--tentativas", type=int, default=12)
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber",
                    help="função de perda [padrão: huber]")
    a = ap.parse_args()

    global N_EPOCAS_MAX
    N_EPOCAS_MAX = a.epocas
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    dados = Path(a.dados)
    print("=" * 68)
    print("  MLP (Keras) — REGRESSÃO de log1p(VL_FOB)")
    print("=" * 68)

    # ---------- dados ----------
    print(f"\n[1/5] Carregando dados (máx. {a.max_por_ano or 'todos'} linhas/ano)...")
    df = carregar_dados(dados, a.max_por_ano, a.semente)
    tr = df[df.particao == "treino"].reset_index(drop=True)
    va = df[df.particao == "validacao"].reset_index(drop=True)
    te = df[df.particao == "teste"].reset_index(drop=True)

    linhas = []
    for nome, parte in [("treino", tr), ("validacao", va), ("teste", te)]:
        for ano, g in parte.groupby("CO_ANO"):
            linhas.append({"particao": nome, "ano": int(ano),
                           "n_linhas": len(g), "n_sh4": g.SH4.nunique()})
    pd.DataFrame(linhas).to_csv(outdir / "particoes.csv", index=False)

    # distribuição do alvo por partição (sem estourar a memória)
    dist = []
    for nome, parte in [("treino", tr), ("validacao", va), ("teste", te)]:
        y = np.log1p(parte.VL_FOB.to_numpy(np.float64))
        dist.append({"particao": nome, "n": len(y),
                     "media_log": y.mean(), "desvio_log": y.std(),
                     "min_log": y.min(), "p50_log": np.quantile(y, .5),
                     "p90_log": np.quantile(y, .9), "max_log": y.max(),
                     "fob_medio_usd": np.expm1(y).mean(),
                     "fob_mediano_usd": np.expm1(np.quantile(y, .5))})
    pd.DataFrame(dist).to_csv(outdir / "distribuicao_alvo.csv", index=False)

    print(f"\n  Treino: {len(tr):,} | Validação: {len(va):,} | Teste: {len(te):,} linhas")
    print(f"  SH4 distintos (treino): {tr.SH4.nunique()} | "
          f"países: {tr.CO_PAIS.nunique()} | UFs: {tr.SG_UF_MUN.nunique()}")

    # ---------- pré-processamento (só no treino) ----------
    print("\n[2/5] Pré-processamento (ajustado apenas no treino)...")
    pre = Preprocessador().fit(tr)
    X_tr = pre.transform_X(tr); y_tr = pre.transform_y(tr)
    X_va = pre.transform_X(va); va_y = pre.transform_y(va)
    X_te = pre.transform_X(te); te_y = pre.transform_y(te)
    # alvo real (escala log1p e US$) para as métricas finais
    te_log = np.log1p(te.VL_FOB.to_numpy(np.float64))
    te_usd = te.VL_FOB.to_numpy(np.float64)
    va_log = np.log1p(va.VL_FOB.to_numpy(np.float64))
    del tr, va, te, df
    n_feat = X_tr.shape[1]
    print(f"  Dimensão de entrada: {n_feat} features "
          f"({len(pre.sh4)} SH4 + {len(pre.paises)} países + {len(pre.ufs)} UFs + 5)")
    print(f"  Alvo: log1p(VL_FOB) padronizado | mu={pre.y_mu:.3f} sd={pre.y_sd:.3f}")
    print(f"  TensorFlow {tf.__version__} | Keras {keras.__version__}")

    # ---------- baselines ingênuos (referência) ----------
    # (a) preditor constante = média do treino; (b) regressão linear
    #     fechada usando apenas log1p(KG_LIQUIDO)
    y_tr_log = pre.despadronizar(y_tr)
    media_tr = float(y_tr_log.mean())
    # coluna de logkg padronizado: penúltima da matriz de features
    logkg_va = X_va[:, -2] * pre.logkg_sd + pre.logkg_mu
    logkg_tr = X_tr[:, -2] * pre.logkg_sd + pre.logkg_mu
    logkg_te = X_te[:, -2] * pre.logkg_sd + pre.logkg_mu
    A = np.vstack([np.ones_like(logkg_tr), logkg_tr]).T
    coef, *_ = np.linalg.lstsq(A, y_tr_log, rcond=None)

    # ---------- busca Optuna ----------
    print(f"\n[3/5] Busca Optuna: {a.tentativas} tentativas, "
          f"loss={a.loss}, métrica = MAE de validação (minimizar)...")

    def objetivo(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        n_ocultas = trial.suggest_int("n_ocultas", 1, 3)
        largura = trial.suggest_categorical("largura", [64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        l2 = trial.suggest_float("l2", 1e-6, 1e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [256, 512, 1024])
        # delta do Huber em unidades do alvo padronizado: 1σ ≈ 1 desvio-
        # padrão de log1p(VL_FOB) (~3,2). 0,1σ já ignora a cauda; 3σ
        # converge para o MSE puro. Busca log em [0,1; 3].
        delta = (trial.suggest_float("huber_delta", 0.1, 3.0, log=True)
                 if a.loss == "huber" else None)
        tf.keras.backend.clear_session()
        modelo = construir_modelo(n_feat, n_ocultas, largura, dropout, l2, lr,
                                  loss_tipo=a.loss, huber_delta=delta or 1.0)
        cb = CallbackoValidacao(X_va, va_y, pre.y_sd, pre, PACIENCIA, "",
                                trial=trial)
        modelo.fit(X_tr, y_tr, validation_data=(X_va, va_y),
                   epochs=N_EPOCAS_MAX, batch_size=batch,
                   callbacks=[cb], verbose=0)
        if cb.podado:
            raise optuna.TrialPruned()
        return cb.melhor_mae

    estudo = optuna.create_study(
        study_name=f"mlp_reg_keras_{a.loss}", direction="minimize",
        sampler=TPESampler(seed=a.semente),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=3),
        storage=f"sqlite:///{outdir / 'optuna.db'}", load_if_exists=True)
    estudo.optimize(objetivo, n_trials=a.tentativas, show_progress_bar=False)

    resumo = estudo.trials_dataframe(
        attrs=("number", "state", "value", "duration", "params"))
    resumo = resumo.rename(columns={"value": "mae_validacao",
                                    "duration": "segundos"})
    cols_busca = ["lr", "n_ocultas", "largura", "dropout", "l2", "batch_size"]
    if a.loss == "huber":
        cols_busca.append("huber_delta")
    for c in cols_busca:
        resumo[c] = resumo.get("params_" + c)
    resumo = resumo[["number", "state", "mae_validacao", "segundos"]
                    + cols_busca]
    resumo.to_csv(outdir / "resultados_busca.csv", index=False)

    completos = sorted(
        (t for t in estudo.trials
         if t.state == optuna.trial.TrialState.COMPLETE),
        key=lambda t: t.value)[:3]                 # direction=minimize
    print("\n  Top-3 da busca (menor MAE de validação):")
    for t in completos:
        print(f"    trial {t.number}: MAE validação = {t.value:.4f} | {t.params}")

    # ---------- retreino top-3 + baseline sem regularização ----------
    print("\n[4/5] Retreinando top-3 + baseline sem regularização...")
    candidatos = [(f"top{t.number}", t.params, a.semente + t.number)
                  for t in completos]
    base = dict(completos[0].params); base["dropout"] = 0.0; base["l2"] = 0.0
    candidatos.append(("baseline_sem_reg", base, a.semente))

    historico, testes = [], {}
    predicoes = {}                                # pred_log por modelo
    melhor_nome = None
    for nome, p, semente in candidatos:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(semente)
        modelo = construir_modelo(n_feat, int(p["n_ocultas"]),
                                  int(p["largura"]), p["dropout"], p["l2"],
                                  p["lr"], loss_tipo=a.loss,
                                  huber_delta=p.get("huber_delta", 1.0))
        t0 = time.perf_counter()
        cb = treinar_modelo(modelo, X_tr, y_tr, X_va, va_y, pre, nome,
                            semente, batch_size=int(p["batch_size"]))
        tempo_treino = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_std = modelo(X_te, training=False).numpy().ravel()
        t_inf = time.perf_counter() - t0
        pred_log = pre.despadronizar(pred_std)
        pred_usd = np.expm1(pred_log)
        m = metricas_reg(te_log, pred_log, te_usd, pred_usd)
        m.update({
            "arquitetura": " -> ".join(
                map(str, [n_feat] + [int(p["largura"])] * int(p["n_ocultas"]))
                ) + " -> 1",
            "parametros": int(modelo.count_params()),
            "dropout": p["dropout"], "l2": p["l2"], "lr": p["lr"],
            "batch_size": p["batch_size"],
            **({"huber_delta": p["huber_delta"]} if "huber_delta" in p else {}),
            "epocas_executadas": len(cb.historico),
            "melhor_epoca": int(np.argmin(
                [h["mae_validacao"] for h in cb.historico]) + 1),
            "mae_validacao_log": float(cb.melhor_mae),
            "segundos_treino": round(tempo_treino, 1),
            "ms_inferencia_por_mil": round(t_inf / len(X_te) * 1000, 3),
        })
        # validação também no teste comparável
        testes[nome] = m
        predicoes[nome] = pred_log
        historico.extend(cb.historico)
        print(f"  {nome}: teste MAE={m['mae_log']:.4f} RMSE={m['rmse_log']:.4f} "
              f"R²={m['r2_log']:.4f} | erro mediano={m.get('erro_pct_mediano', float('nan')):.1f}% "
              f"| {m['parametros']:,} parâmetros | treino {tempo_treino:.0f}s")
        if melhor_nome is None or m["mae_validacao_log"] < testes[melhor_nome]["mae_validacao_log"]:
            melhor_nome = nome

    # baselines ingênuos no teste
    m_media = metricas_reg(te_log, np.full_like(te_log, media_tr),
                           te_usd, np.expm1(np.full_like(te_log, media_tr)))
    m_linear = metricas_reg(te_log, coef[0] + coef[1] * logkg_te,
                            te_usd, np.expm1(coef[0] + coef[1] * logkg_te))
    baselines = {
        "media_treino": {"descricao": "prediz sempre a média de log1p(VL_FOB) "
                                      "observada no treino",
                         **m_media},
        "linear_somente_peso": {"descricao": "reta fechada log1p(VL_FOB) ~ "
                                             "log1p(KG_LIQUIDO) ajustada no treino",
                                "coeficientes": [float(coef[0]), float(coef[1])],
                                **m_linear},
    }

    pd.DataFrame(historico).to_csv(outdir / "historico_treino.csv", index=False)

    melhor_val = min(testes, key=lambda k: testes[k]["mae_validacao_log"])
    melhor_pred_log = predicoes[melhor_val]
    pd.DataFrame({"y_real_log": te_log,
                  "y_predito_log": melhor_pred_log,
                  "y_real_usd": te_usd,
                  "y_predito_usd": np.expm1(melhor_pred_log)}).to_csv(
        outdir / "predicoes_teste.csv", index=False)

    saida = {
        "implementacao": f"Keras {keras.__version__} / TensorFlow {tf.__version__}",
        "problema": {"tipo": "regressão", "alvo": "log1p(VL_FOB)",
                     "entradas": ["SH4 (one-hot)", "CO_PAIS (one-hot)",
                                  "SG_UF_MUN (one-hot)", "CO_MES (seno/cosseno)",
                                  "CO_ANO (padronizado)",
                                  "log1p(KG_LIQUIDO) (padronizado)",
                                  "indicador de peso ausente/zero"],
                     "unidade_observacao": "agregado município × SH4 × país × mês"},
        "protocolo": {"treino": "2015-2023", "validacao": "2024",
                      "teste": "2025", "max_por_ano": a.max_por_ano,
                      "semente": a.semente,
                      "amostras": {"treino": len(X_tr), "validacao": len(X_va),
                                   "teste": len(X_te)},
                      "criterio_parada": f"parada antecipada (paciencia={PACIENCIA}, "
                                         "métrica=MAE validação em log1p)",
                      "perda": (f"Huber escrito à mão (delta buscado no Optuna; "
                                "unidade do delta = desvio-padrão do alvo "
                                "padronizado)"
                                if a.loss == "huber" else
                                "MSE sobre o alvo padronizado "
                                "(mu/sd ajustados no treino)")},
        "busca": {"ferramenta": "Optuna TPE + MedianPruner",
                  "tentativas": a.tentativas,
                  "loss": a.loss,
                  "metrica": "MAE de validação (escala log1p, minimizar)",
                  "melhor_trial": int(estudo.best_trial.number),
                  "melhor_valor_validacao": float(estudo.best_trial.value)},
        "melhor_modelo": melhor_val,
        "baselines": baselines,
        "modelos_teste": testes,
    }
    (outdir / "metricas_teste.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- figuras ----------
    print("\n[5/5] Gerando figuras...")
    hist_df = pd.DataFrame(historico)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    h = hist_df[hist_df.modelo == melhor_val]
    ax[0, 0].plot(h.epoca, h.perda_treino, label="treino")
    ax[0, 0].plot(h.epoca, h.perda_validacao, label="validação")
    ax[0, 0].set_title(f"Perda ({'Huber' if a.loss == 'huber' else 'MSE'}, "
                       f"alvo padronizado) — {melhor_val}")
    ax[0, 0].set_xlabel("época"); ax[0, 0].legend()
    for nome, g in hist_df.groupby("modelo"):
        ax[0, 1].plot(g.epoca, g.mae_validacao, label=nome)
    ax[0, 1].set_title("MAE de validação (log1p) por época")
    ax[0, 1].set_xlabel("época"); ax[0, 1].legend()

    nomes = list(testes)
    vals = [testes[n]["mae_log"] for n in nomes]
    barras = ax[1, 0].bar(nomes + ["baseline_média", "baseline_linear"],
                          vals + [baselines["media_treino"]["mae_log"],
                                  baselines["linear_somente_peso"]["mae_log"]],
                          color=["tab:blue" if n == melhor_val else "tab:gray"
                                 for n in nomes] + ["tab:olive", "tab:olive"])
    ax[1, 0].set_title("MAE no teste (2025, escala log1p) — menor é melhor")
    ax[1, 0].tick_params(axis="x", rotation=20)

    n_pts = min(4000, len(te_log))
    amostra = np.random.default_rng(0).choice(len(te_log), n_pts, replace=False)
    ax[1, 1].scatter(te_log[amostra], melhor_pred_log[amostra], s=4, alpha=.35)
    lo = min(te_log.min(), melhor_pred_log.min())
    hi = max(te_log.max(), melhor_pred_log.max())
    ax[1, 1].plot([lo, hi], [lo, hi], "r--", lw=1)
    ax[1, 1].set_xlabel("real (log1p US$)"); ax[1, 1].set_ylabel("predito")
    ax[1, 1].set_title(f"Predito × real — {melhor_val} (teste, amostra)")
    fig.tight_layout()
    fig.savefig(outdir / "curvas_treinamento.png")
    plt.close(fig)

    best = testes[melhor_val]
    print("\n" + "=" * 68)
    print(f"  CONCLUSÃO — melhor modelo (por validação): {melhor_val}")
    print(f"  Validação MAE: {best['mae_validacao_log']:.4f} (log1p)")
    print(f"  Teste MAE={best['mae_log']:.4f} | RMSE={best['rmse_log']:.4f} | "
          f"R²={best['r2_log']:.4f}")
    print(f"  Erro mediano no US$: {best['erro_pct_mediano']:.1f}% | "
          f"P90: {best['erro_pct_p90']:.1f}%")
    print(f"  Baselines — média: R²={baselines['media_treino']['r2_log']:.4f} "
          f"MAE={baselines['media_treino']['mae_log']:.4f} | "
          f"só-peso: R²={baselines['linear_somente_peso']['r2_log']:.4f} "
          f"MAE={baselines['linear_somente_peso']['mae_log']:.4f}")
    print(f"  Saídas em: {outdir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
