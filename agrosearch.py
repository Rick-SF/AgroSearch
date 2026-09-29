"""
AgroSearch - Motor de Busca Inteligente
UNIPÊ · Tópicos Avançados (Recuperação de Informação / PLN) · Laboratório 04
Aluno: Ricardo Silva Flores - Ciência da Computação

Como executar:
    pip install streamlit pandas
    streamlit run agrosearch.py

Todo o processamento (pré-processamento, índice invertido, TF, IDF, TF-IDF e
similaridade de cosseno) foi escrito do zero, usando apenas a biblioteca
padrão do Python. O pandas serve somente para montar as tabelas exibidas
na interface.
"""

import math
import string
import unicodedata

import pandas as pd
import streamlit as st

# =============================================================================
# BASE DE DOCUMENTOS
# =============================================================================
CORPUS = {
    1: "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    2: "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    3: "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    4: "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    5: "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}

# =============================================================================
# FASE 1 - PRÉ-PROCESSAMENTO
# =============================================================================
# Lista de stopwords escrita já sem acento, porque a remoção acontece
# depois da normalização.
PALAVRAS_VAZIAS = frozenset(
    """
    a o e as os um uma uns umas de do da dos das em no na nos nas
    ao aos à às por pelo pela pelos pelas para pra com sem sob sobre
    que se ou mas nem ja nao sim muito muita pouco mais menos tambem
    ser estar ter ha foi era sao esta estao tem pode podem deve
    feito feita durante ate apos entre como quando onde qual quais
    este esta isto esse essa isso aquele aquela seu sua seus suas
    """.split()
)


def tokenizacao(texto):
    """Etapa 1: separa o texto em palavras, trocando pontuação por espaço."""
    tabela = str.maketrans({p: " " for p in string.punctuation})
    return texto.translate(tabela).split()


def normalizacao(token):
    """Etapa 2: caixa baixa e remoção de acentos (decomposição NFKD)."""
    token = token.lower()
    sem_acento = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in sem_acento if not unicodedata.combining(ch))


def eh_stopword(token):
    """Etapa 3: verifica se o token está na lista de palavras vazias."""
    return token in PALAVRAS_VAZIAS


# Regras do stemmer em etapas, no estilo do RSLP (Orengo & Huyck, 2001),
# mas numa versão reduzida feita para este trabalho.
# Cada regra: (sufixo, tamanho mínimo do radical, substituição)
REGRAS_PLURAL = [
    ("oes", 2, "ao"), ("aes", 2, "ao"), ("ais", 2, "al"),
    ("eis", 2, "el"), ("ns", 1, "m"), ("is", 2, "il"), ("s", 2, ""),
]
REGRAS_NOMINAIS = [
    ("amento", 3, ""), ("imento", 3, ""), ("adora", 3, ""), ("ador", 3, ""),
    ("acao", 3, ""), ("idade", 3, ""), ("ivel", 3, ""), ("mente", 4, ""),
    ("ismo", 3, ""), ("ista", 3, ""), ("ico", 3, ""), ("ica", 3, ""),
    ("oso", 3, ""), ("osa", 3, ""), ("agem", 3, ""), ("ura", 4, ""),
]
REGRAS_VERBAIS = [
    ("ando", 2, ""), ("endo", 3, ""), ("indo", 3, ""),
    ("aram", 2, ""), ("eram", 3, ""), ("ava", 2, ""),
    ("ar", 2, ""), ("er", 2, ""), ("ir", 3, ""), ("am", 2, ""), ("em", 2, ""),
]
VOGAIS_FINAIS = "aeo"


def _aplica_regras(palavra, regras):
    """Aplica a primeira regra compatível e informa se houve troca."""
    for sufixo, minimo, troca in regras:
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= minimo:
            return palavra[: len(palavra) - len(sufixo)] + troca, True
    return palavra, False


def stemming(palavra):
    """Etapa 4: reduz a palavra ao radical.
    Ordem: plural -> sufixo nominal -> (se não houve) sufixo verbal -> vogal final.
    """
    if len(palavra) <= 3:
        return palavra
    palavra, _ = _aplica_regras(palavra, REGRAS_PLURAL)
    palavra, mudou = _aplica_regras(palavra, REGRAS_NOMINAIS)
    if not mudou:
        palavra, _ = _aplica_regras(palavra, REGRAS_VERBAIS)
    if len(palavra) > 3 and palavra[-1] in VOGAIS_FINAIS and not palavra.endswith("ao"):
        palavra = palavra[:-1]
    return palavra


def pipeline(texto, remover_stopwords=True, aplicar_stemming=True, etapas=False):
    """Roda as 4 etapas. Com etapas=True devolve também o resultado parcial de cada uma."""
    tokens = tokenizacao(texto)
    normalizados = [normalizacao(t) for t in tokens]
    filtrados = [t for t in normalizados if not eh_stopword(t)] if remover_stopwords else list(normalizados)
    radicais = [stemming(t) for t in filtrados] if aplicar_stemming else list(filtrados)
    if etapas:
        return radicais, {
            "Tokenização": tokens,
            "Normalização": normalizados,
            "Sem stopwords": filtrados,
            "Stemming": radicais,
        }
    return radicais


# =============================================================================
# FASE 2 - ÍNDICE INVERTIDO
# =============================================================================
def indexar(docs_processados):
    """Monta o índice invertido {termo: [ids dos documentos]}.
    Também guarda as frequências {termo: {doc_id: f(t,d)}}, que o TF usa depois.
    """
    postings = {}
    frequencias = {}
    for doc_id, termos in docs_processados.items():
        for termo in termos:
            freq_termo = frequencias.setdefault(termo, {})
            freq_termo[doc_id] = freq_termo.get(doc_id, 0) + 1
    for termo in sorted(frequencias):
        postings[termo] = sorted(frequencias[termo])
    return postings, frequencias


# =============================================================================
# FASE 3 - TF, IDF, TF-IDF E RANQUEAMENTO
# =============================================================================
def tf(termo, doc_id, frequencias, tamanhos):
    """TF(t,d) = f(t,d) / |d|, onde |d| é o total de termos do documento."""
    if tamanhos[doc_id] == 0:
        return 0.0
    return frequencias.get(termo, {}).get(doc_id, 0) / tamanhos[doc_id]


def idf(termo, postings, total_docs):
    """IDF(t) = log10(N / df(t)). Termo que não está no índice recebe 0."""
    df = len(postings.get(termo, []))
    return math.log10(total_docs / df) if df else 0.0


def vetor_documento(doc_id, termos, postings, frequencias, tamanhos, n):
    """Vetor TF-IDF (esparso) de um documento."""
    return {t: tf(t, doc_id, frequencias, tamanhos) * idf(t, postings, n) for t in set(termos)}


def vetor_consulta(termos_q, postings, n):
    """Vetor TF-IDF da consulta, com o mesmo TF relativo usado nos documentos."""
    if not termos_q:
        return {}
    vetor = {}
    for t in set(termos_q):
        vetor[t] = (termos_q.count(t) / len(termos_q)) * idf(t, postings, n)
    return vetor


def cosseno(u, v):
    """Bônus: cos(q,d) = (q · d) / (|q| · |d|)."""
    produto = sum(peso * v.get(t, 0.0) for t, peso in u.items())
    norma_u = math.sqrt(sum(p * p for p in u.values()))
    norma_v = math.sqrt(sum(p * p for p in v.values()))
    return produto / (norma_u * norma_v) if norma_u and norma_v else 0.0


def ranquear(termos_q, docs_processados, postings, frequencias):
    """Score(d) = soma, para cada termo distinto t da consulta, de TF(t,d) x IDF(t).
    Só entram na conta os documentos candidatos (os que aparecem nas listas
    do índice invertido para algum termo da consulta).
    """
    n = len(docs_processados)
    tamanhos = {d: len(ts) for d, ts in docs_processados.items()}
    termos_unicos = list(dict.fromkeys(termos_q))

    candidatos = set()
    for t in termos_unicos:
        candidatos.update(postings.get(t, []))

    q_vec = vetor_consulta(termos_q, postings, n)
    calculos = []
    ranking = []
    for doc_id in sorted(candidatos):
        acumulado = 0.0
        for t in termos_unicos:
            v_tf = tf(t, doc_id, frequencias, tamanhos)
            v_idf = idf(t, postings, n)
            acumulado += v_tf * v_idf
            calculos.append({"Doc": doc_id, "Termo": t, "TF": v_tf, "IDF": v_idf, "TF x IDF": v_tf * v_idf})
        d_vec = vetor_documento(doc_id, docs_processados[doc_id], postings, frequencias, tamanhos, n)
        ranking.append({"Doc": doc_id, "TF-IDF acumulado": acumulado, "Cosseno": cosseno(q_vec, d_vec)})

    ranking.sort(key=lambda r: (-r["TF-IDF acumulado"], r["Doc"]))
    return ranking, calculos


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================
def destacar_vencedor(linha):
    cor = "background-color: #d8f3dc; font-weight: bold" if linha.name == 0 else ""
    return [cor] * len(linha)


def app():
    st.set_page_config(page_title="AgroSearch", layout="wide")
    st.title("AgroSearch")
    st.write("Busca textual nos manuais técnicos da AgroTech Solutions com índice invertido e TF-IDF.")

    st.sidebar.subheader("Pré-processamento")
    usar_stop = st.sidebar.checkbox("Remover stopwords", value=True)
    usar_stem = st.sidebar.checkbox("Aplicar stemming", value=True)
    st.sidebar.subheader("Ranqueamento")
    criterio = st.sidebar.radio("Ordenar resultados por", ["TF-IDF acumulado", "Cosseno (bônus)"])
    st.sidebar.divider()
    st.sidebar.caption(f"Base com {len(CORPUS)} documentos")

    processados = {d: pipeline(txt, usar_stop, usar_stem) for d, txt in CORPUS.items()}
    postings, frequencias = indexar(processados)
    n = len(CORPUS)

    fase1, fase2, fase3 = st.tabs(["Fase 1 - Pré-processamento", "Fase 2 - Índice invertido", "Fase 3 - Busca"])

    # ---------- Fase 1 ----------
    with fase1:
        total_tokens = sum(len(t) for t in processados.values())
        a, b = st.columns(2)
        a.metric("Termos únicos no vocabulário", len(postings))
        b.metric("Total de tokens na base", total_tokens)

        linhas = []
        for d, txt in CORPUS.items():
            _, passos = pipeline(txt, usar_stop, usar_stem, etapas=True)
            linhas.append({"Doc": d, **{k: " ".join(v) for k, v in passos.items()}})
        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)

        st.write("**Vocabulário atual:**")
        st.code(" | ".join(postings.keys()), language=None)

    # ---------- Fase 2 ----------
    with fase2:
        formato = st.radio("Exibir como", ["Tabela", "JSON"], horizontal=True)
        if formato == "JSON":
            st.json(postings)
        else:
            tabela = pd.DataFrame(
                [{"Termo": t, "Documentos": ids, "df": len(ids), "IDF": round(idf(t, postings, n), 4)}
                 for t, ids in postings.items()]
            )
            st.dataframe(tabela, hide_index=True, use_container_width=True)

    # ---------- Fase 3 ----------
    with fase3:
        consulta = st.text_input("Digite a consulta", placeholder="ex.: lagartas na soja")
        st.latex(r"TF(t,d)=\frac{f(t,d)}{|d|}\qquad IDF(t)=\log_{10}\frac{N}{df(t)}\qquad "
                 r"Score(d)=\sum_{t\in q}TF(t,d)\cdot IDF(t)")

        if not consulta.strip():
            st.info("Digite uma consulta para ver o ranking.")
            return

        termos_q = pipeline(consulta, usar_stop, usar_stem)
        st.write("Termos da consulta após o pipeline:", termos_q or "nenhum")
        fora = [t for t in dict.fromkeys(termos_q) if t not in postings]
        if fora:
            st.warning("Não estão no vocabulário: " + ", ".join(fora))

        ranking, calculos = ranquear(termos_q, processados, postings, frequencias)
        if criterio.startswith("Cosseno"):
            ranking.sort(key=lambda r: (-r["Cosseno"], -r["TF-IDF acumulado"], r["Doc"]))
        if not ranking or ranking[0]["TF-IDF acumulado"] == 0:
            st.error("Nenhum documento relevante para essa consulta.")
            return

        vencedor = ranking[0]
        st.success(f"Documento mais relevante: Doc {vencedor['Doc']} - {CORPUS[vencedor['Doc']]}")

        df_rank = pd.DataFrame(
            [{"Posição": i, "Doc": r["Doc"], "TF-IDF acumulado": round(r["TF-IDF acumulado"], 4),
              "Cosseno": round(r["Cosseno"], 4), "Trecho": CORPUS[r["Doc"]]}
             for i, r in enumerate(ranking, start=1)]
        )
        st.dataframe(df_rank.style.apply(destacar_vencedor, axis=1).format(precision=4),
                     hide_index=True, use_container_width=True)

        with st.expander("Ver TF, IDF e TF-IDF de cada termo por documento"):
            st.dataframe(pd.DataFrame(calculos).round(4), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    app()
