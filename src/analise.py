#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
  EDA — Exportação brasileira por Município e SH4 (Secex/MDIC)
  Disciplina: Deep Learning — Classificação do capítulo SH2
=====================================================================
Uso:
    python analise.py                          # usa ../datasets/EXP_2025_MUN.csv
    python analise.py --csv outro.csv --out ./eda_out

O script gera:
  - relatório textual completo (stdout + relatorio_eda.txt)
  - figuras PNG em --out
  - tabelas resumo em CSV (para colar no relatório da disciplina)
=====================================================================
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


# ----------------------------------------------------------------------
# 0. CONFIGURAÇÃO
# ----------------------------------------------------------------------
COLS = ["CO_ANO", "CO_MES", "SH4", "CO_PAIS", "SG_UF_MUN", "CO_MUN",
        "KG_LIQUIDO", "VL_FOB"]

# Nomes dos capítulos SH2 mais comuns (complemente se desejar)
NOMES_SH2 = {
    1: "Animais vivos", 2: "Carnes e miudezas", 3: "Peixes e crustáceos",
    4: "Leite, laticínios e ovos", 5: "Animais e produtos animais",
    6: "Plantas vivas e flores", 7: "Legumes e hortaliças", 8: "Frutas",
    9: "Café, chá e especiarias", 10: "Cereais", 11: "Produtos de moagem",
    12: "Sementes oleaginosas", 13: "Látex e resinas", 14: "Materiais vegetais",
    15: "Gorduras e óleos", 16: "Preparados cárneos/peixe", 17: "Açúcares",
    18: "Cacau e preparações", 19: "Preparados de cereais", 20: "Preparados hortícolas",
    21: "Preparados alimentícios", 22: "Bebidas e vinagre", 23: "Resíduos alimentares",
    24: "Tabaco", 25: "Sal e minerais", 26: "Minérios", 27: "Combustíveis minerais",
    28: "Produtos químicos inorgânicos", 29: "Produtos orgânicos", 30: "Produtos farmacêuticos",
    31: "Fertilizantes", 32: "Corantes e tintas", 33: "Óleos essenciais e cosméticos",
    34: "Sabões e preparações", 35: "Proteínas e colas", 36: "Explosivos",
    37: "Foto e cinema", 38: "Produtos diversos químicos", 39: "Plásticos",
    40: "Borracha", 41: "Couros e peles", 42: "Obras de couro",
    43: "Peles e pelúcias", 44: "Madeira e obras", 45: "Cortiça",
    46: "Obras de palha", 47: "Polpa e papel", 48: "Papel e papelão",
    49: "Livros e impressos", 50: "Sedas", 51: "Lã e pelos", 52: "Algodão",
    53: "Fibras têxteis", 54: "Fios e fibras sintéticas", 55: "Fios de metal",
    56: "Feltros e não-tecidos", 57: "Tapetes", 58: "Tecidos especiais",
    59: "Tecidos impregnados", 60: "Malhas", 61: "Vestuário de malha",
    62: "Vestuário não-malha", 63: "Outros têxteis", 64: "Calçados",
    65: "Chapéus", 66: "Guarda-chuvas", 67: "Penas e plumas", 68: "Obras de pedra",
    69: "Produtos cerâmicos", 70: "Vidro", 71: "Metais preciosos e pedras",
    72: "Ferro e aço", 73: "Obras de ferro/aço", 74: "Cobre", 75: "Níquel",
    76: "Alumínio", 78: "Chumbo", 79: "Zinco", 80: "Estanho",
    81: "Metais comuns diversos", 82: "Ferramentas", 83: "Utensílios metálicos",
    84: "Máquinas mecânicas", 85: "Máquinas e aparelhos elétricos",
    86: "Veículos ferroviários", 87: "Veículos e peças", 88: "Aeronaves",
    89: "Embarcações", 90: "Instrumentos ópticos/precisão", 91: "Relógios",
    92: "Instrumentos musicais", 93: "Armas e munições", 94: "Móveis",
    95: "Brinquedos", 96: "Manufaturas diversas", 97: "Obras de arte",
}


def carregar(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=";", dtype=str, usecols=COLS)
    for c in ["CO_ANO", "CO_MES"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["KG_LIQUIDO", "VL_FOB"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["CO_PAIS"] = df["CO_PAIS"].str.zfill(3)
    df["CO_MUN"] = df["CO_MUN"].str.zfill(7)
    df["SH4"] = df["SH4"].str.zfill(4)
    df["SH2"] = df["SH4"].str[:2].astype(int)          # <-- variável alvo
    df["PRECO_KG"] = df["VL_FOB"] / df["KG_LIQUIDO"].replace(0, np.nan)
    return df


def imprimir(txt, arq):
    print(txt)
    arq.write(txt + "\n")


# ----------------------------------------------------------------------
# 1. ANÁLISE
# ----------------------------------------------------------------------
def analisar(df: pd.DataFrame, outdir: Path):
    rel = open(outdir / "relatorio_eda.txt", "w", encoding="utf-8")
    L = lambda t="": imprimir(t, rel)

    L("=" * 70)
    L("  RELATÓRIO EDA — EXPORTAÇÃO BRASILEIRA (MUNICÍPIO × SH4)")
    L("=" * 70)
    L(f"\nAmostra: {len(df):,} transações | Ano(s): {sorted(df.CO_ANO.dropna().unique())}"
      f" | Meses: {sorted(df.CO_MES.dropna().unique())}")

    # ---------- 1.1 QUALIDADE ----------
    L("\n" + "-" * 70 + "\n1.1 QUALIDADE DOS DADOS\n" + "-" * 70)
    aus = df.isna().sum()
    L(f"Valores ausentes por coluna:\n{aus[aus > 0].to_string() if (aus > 0).any() else '  Nenhum'}")
    L(f"Duplicadas exatas: {df.duplicated().sum()}")
    chave = ["CO_ANO", "CO_MES", "SH4", "CO_PAIS", "SG_UF_MUN", "CO_MUN"]
    L(f"Duplicadas pela chave natural: {df.duplicated(subset=chave).sum()}")
    n_kg0 = (df.KG_LIQUIDO == 0).sum()
    n_fob0 = (df.VL_FOB == 0).sum()
    n_neg = (df.VL_FOB < 0).sum()
    L(f"KG_LIQUIDO = 0: {n_kg0:,} ({n_kg0/len(df)*100:.2f}%)  <- imputar ou flag")
    L(f"VL_FOB = 0:   {n_fob0:,} ({n_fob0/len(df)*100:.2f}%)")
    L(f"VL_FOB < 0:   {n_neg:,}  (revisar/eliminar)")

    # ---------- 1.2 ALVO SH2 ----------
    L("\n" + "-" * 70 + "\n1.2 DISTRIBUIÇÃO DO ALVO (CAPÍTULO SH2)\n" + "-" * 70)
    vc = df.SH2.value_counts()
    L(f"Classes presentes: {df.SH2.nunique()} de 97 | Razão maior/menor classe: "
      f"{vc.iloc[0]/max(vc.iloc[-1],1):,.0f}:1")
    L(f"Top classe: {vc.index[0]:02d} – {NOMES_SH2.get(vc.index[0],'')} "
      f"({vc.iloc[0]:,} transações, {vc.iloc[0]/len(df)*100:.1f}%)")
    tab = vc.head(20).rename_axis("SH2").reset_index(name="n")
    tab["capitulo"] = tab.SH2.map(lambda s: NOMES_SH2.get(s, ""))
    tab["pct"] = (tab.n / len(df) * 100).round(2)
    L(tab.to_string(index=False))
    tab.to_csv(outdir / "distribuicao_sh2.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    vc.head(20).sort_values().plot.barh(ax=ax, color="#2E86AB")
    ax.set_title("Top 20 capítulos SH2 — frequência")
    ax.set_xlabel("nº de transações")
    plt.tight_layout(); plt.savefig(outdir / "01_dist_alvo_sh2.png"); plt.close()

    # ---------- 1.3 VARIÁVEIS CATEGÓRICAS ----------
    L("\n" + "-" * 70 + "\n1.3 VARIÁVEIS CATEGÓRICAS\n" + "-" * 70)
    L(f"UFs distintas: {df.SG_UF_MUN.nunique()} | Países distintos: {df.CO_PAIS.nunique()} "
      f"| Municípios distintos: {df.CO_MUN.nunique():,}")
    L("\nTop 10 UF por nº de transações:")
    L(df.SG_UF_MUN.value_counts().head(10).to_string())
    L("\nTop 10 países destino (código):")
    L(df.CO_PAIS.value_counts().head(10).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df.SG_UF_MUN.value_counts().head(15).sort_values().plot.barh(ax=axes[0], color="#F18F01")
    axes[0].set_title("Top 15 UFs (nº de transações)")
    df.CO_PAIS.value_counts().head(15).sort_values().plot.barh(ax=axes[1], color="#C73E1D")
    axes[1].set_title("Top 15 países destino (código)")
    plt.tight_layout(); plt.savefig(outdir / "02_categoricas.png"); plt.close()

    # ---------- 1.4 VARIÁVEIS NUMÉRICAS ----------
    L("\n" + "-" * 70 + "\n1.4 VARIÁVEIS NUMÉRICAS (ASSIMETRIA E OUTLIERS)\n" + "-" * 70)
    for col in ["KG_LIQUIDO", "VL_FOB", "PRECO_KG"]:
        s = df[col].dropna()
        L(f"\n{col}: média={s.mean():,.1f} | mediana={s.median():,.1f} | "
          f"máx={s.max():,.1f} | desvio={s.std():,.1f}")
        L(f"  assimetria={stats.skew(s):.2f} | curtose={stats.kurtosis(s):.2f} "
          f"| P95={s.quantile(.95):,.0f} | P99={s.quantile(.99):,.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(np.log10(df.KG_LIQUIDO[df.KG_LIQUIDO > 0]), bins=60, color="#2E86AB")
    axes[0, 0].set_title("log10(KG_LIQUIDO)")
    axes[0, 1].hist(np.log10(df.VL_FOB[df.VL_FOB > 0]), bins=60, color="#A23B72")
    axes[0, 1].set_title("log10(VL_FOB)")
    pos = df[(df.KG_LIQUIDO > 0) & (df.VL_FOB > 0)]
    axes[1, 0].scatter(np.log10(pos.KG_LIQUIDO), np.log10(pos.VL_FOB),
                       s=3, alpha=0.25, color="#3B1F2B")
    axes[1, 0].set_xlabel("log10(KG)"); axes[1, 0].set_ylabel("log10(US$)")
    axes[1, 0].set_title("Peso × Valor (escala log-log)")
    ax = axes[1, 1]
    top = vc.head(8).index
    dados = [np.log10(df.loc[df.SH2 == c, "PRECO_KG"].dropna()) for c in top]
    ax.boxplot(dados, tick_labels=[f"{c:02d}" for c in top])
    ax.set_title("log10(preço/kg) por capítulo SH2 (top 8)")
    plt.tight_layout(); plt.savefig(outdir / "03_numericas.png"); plt.close()

    # ---------- 1.5 SAZONALIDADE ----------
    L("\n" + "-" * 70 + "\n1.5 SAZONALIDADE (MÊS)\n" + "-" * 70)
    mensal = df.groupby("CO_MES").agg(n=("VL_FOB", "size"), fob=("VL_FOB", "sum"),
                                      kg=("KG_LIQUIDO", "sum"))
    L(mensal.round(0).to_string())
    mensal.to_csv(outdir / "mensal.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(mensal.index.astype(int), mensal.fob / 1e6, color="#2E86AB")
    ax.set_xlabel("mês"); ax.set_ylabel("US$ milhões")
    ax.set_title("Valor FOB total por mês")
    plt.tight_layout(); plt.savefig(outdir / "04_sazonalidade.png"); plt.close()

    # ---------- 1.6 OUTLIERS MULTIVARIADOS ----------
    L("\n" + "-" * 70 + "\n1.6 OUTLIERS MULTIVARIADOS (PREÇO/KG)\n" + "-" * 70)
    p99 = df.PRECO_KG.quantile(0.999)
    ext = df[df.PRECO_KG > p99].copy()
    ext["capitulo"] = ext.SH2.map(lambda s: NOMES_SH2.get(s, ""))
    L(f"Top 0,1% preço/kg (> US$ {p99:,.0f}/kg): {len(ext):,} transações")
    L(ext[["CO_MES", "SH4", "capitulo", "SG_UF_MUN", "KG_LIQUIDO", "VL_FOB", "PRECO_KG"]]
      .sort_values("PRECO_KG", ascending=False).head(10).to_string(index=False))
    ext.to_csv(outdir / "outliers_preco_kg.csv", index=False)

    # ---------- 1.7 RELAÇÃO UF × CLASSE (prova do padrão aprendível) ----------
    L("\n" + "-" * 70 + "\n1.7 CONCENTRAÇÃO UF × CAPÍTULO (por que o modelo aprende)\n" + "-" * 70)
    ct = pd.crosstab(df.SG_UF_MUN, df.SH2, normalize="index") * 100
    for uf in ct.sum(axis=1).sort_values(ascending=False).head(6).index:
        top3 = ct.loc[uf].nlargest(3)
        L(f"  {uf}: " + " | ".join(f"cap.{c:02d} {v:.1f}%" for c, v in top3.items()))
    ct.to_csv(outdir / "uf_por_sh2_pct.csv")

    L("\nFIM DO RELATÓRIO")
    rel.close()
    print(f"\n✔ EDA concluída. Resultados em: {outdir}")


def main():
    # Caminhos padrão relativos ao projeto (raiz = pasta pai de src/)
    raiz = Path(__file__).resolve().parent.parent
    csv_padrao = raiz / "datasets" / "EXP_2025_MUN.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(csv_padrao),
                    help=f"arquivo CSV da Secex (sep=';') [padrão: {csv_padrao}]")
    ap.add_argument("--out", default=str(raiz / "relatorios" / "eda_out"),
                    help="pasta de saída")
    ap.add_argument("--amostra", type=int, default=None,
                    help="nº máx. de linhas (para testes rápidos)")
    ap.add_argument("--semente", type=int, default=42)
    a = ap.parse_args()

    if not Path(a.csv).exists():
        sys.exit(f"[erro] CSV não encontrado: {a.csv}")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    df = carregar(a.csv)
    if a.amostra and len(df) > a.amostra:
        df = df.sample(a.amostra, random_state=a.semente)
        print(f"[info] usando amostra aleatória de {a.amostra:,} linhas")
    analisar(df, outdir)


if __name__ == "__main__":
    main()