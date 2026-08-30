"""
RETRIEVAL-AUGMENTED GENERATION  (runs on every question)

Phase 1 pipeline (default, hybrid=False):
    {"question", "k"}
      |-> .assign(hits=...)     STEP R: RETRIEVAL via Chroma vector search
      |-> .assign(context=...)  STEP F: FORMAT numbered blocks with source+page
      |-> .assign(response=...) STEP G: GENERATE via Groq LLM

Phase 2 pipeline (hybrid=True):
    Same F and G steps, but STEP R is replaced by retriever.hybrid_retrieve():
        BM25 (20) + vector (20) -> RRF fusion -> cross-encoder top-5

    {"question", "k"}                       <- what you pass in
      |-> .assign(hits=...)                 STEP R: RETRIEVAL
              embed the question with the SAME Gemini model used at ingest,
              then ask Chroma for the k nearest chunk vectors (cosine).
              Result: list of hit dicts {text, metadata{source,page}, score}
      |-> .assign(context=...)              STEP F: FORMATTING
              number the hits [1]..[n] and prefix each with its source+page,
              so the model can cite like "[1]".
      |-> .assign(response=...)             STEP G: GENERATION
              prompt template -> ChatOpenAI(Groq) -> string parser.

`.assign` threads a growing dict through the steps: after step R the dict
gains "hits", after F it gains "context", after G it gains "response".
"""

from langchain_core.output_parsers import StrOutputParser    # AIMessage -> str
from langchain_core.prompts import ChatPromptTemplate        # fill {placeholders}
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI                      # talks to Groq too

from config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    TOP_K,
)
from ingest import get_embeddings, create_chroma   # reuse the SAME embedding model as ingest!
# ^ critical: query vectors must live in the same "space" as chunk vectors


SYSTEM_RULES = (
    # Grounding rules: without this the model would answer from its own
    # training knowledge instead of your documents (= hallucination).
    "You are a precise research assistant. Answer ONLY from the numbered context "
    "blocks provided. Cite sources inline as [1], [2], ... matching the block you "
    "used. If the context does not contain the answer, reply exactly: "
    "'I could not find this in the documents.'"
)

# Prompt template: LangChain fills {context} and {question} at runtime.
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_RULES),
        ("human", "Context blocks:\n\n{context}\n\nQuestion: {question}\n"
                  "Answer with inline citations like [1]."),
    ]
)

# The generator. ChatOpenAI is OpenAI's client class, but because we set
# base_url=GROQ_BASE_URL every request goes to Groq's servers instead.
_llm = ChatOpenAI(
    base_url=GROQ_BASE_URL,
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.2,   # low = factual & repeatable, high = creative
)


def _build_context(d: dict) -> str:
    """
    STEP F - turn retrieved hits into the text block shown to the model.

    Produces:
        [1] Source: sample_notes.txt | Page 1
        <full chunk text>
        [2] Source: paper.pdf | Page 8
        <...>
    The [n] labels are what the model references in its citations.
    """
    blocks = [
        f"[{i}] Source: {h['metadata'].get('source', '?')} "
        f"| Page {h['metadata'].get('page', '?')}\n{h['text']}"
        for i, h in enumerate(d["hits"], start=1)
    ]
    return "\n\n".join(blocks)


def _retrieve(d: dict) -> list[dict]:
    """
    STEP R - SEMANTIC SEARCH.

    d = {"question": "...", "k": 5}

    similarity_search_with_score():
      1. embeds d["question"] via get_embeddings()  (one Gemini API call)
      2. compares that vector against EVERY stored chunk vector (cosine)
      3. returns the k closest pairs (Document, distance)

    distance = how far apart the vectors are (0 = identical),
    so we store score = 1 - distance, i.e. SIMILARITY (1 = identical).
    """
    vs = create_chroma()
    if len(vs.get(limit=1)["ids"]) == 0:
        raise RuntimeError("Vector store is empty - run: python main.py ingest")

    pairs = vs.similarity_search_with_score(d["question"], k=d.get("k", TOP_K))
    return [
        {
            "text": doc.page_content,            # the chunk itself
            "metadata": dict(doc.metadata),      # {"source": ..., "page": ...}
            "score": round(1 - dist, 4),         # cosine similarity, higher=better
        }
        for doc, dist in pairs
    ]


# ---------------------------------------------------------------------------
# THE CHAIN
# _respond is a mini-chain: formatted prompt -> Groq -> plain string.
# rag_chain glues retrieval + formatting + generation into ONE callable.
# Each RunnablePassthrough.assign(...) adds one key to the running dict.
# ---------------------------------------------------------------------------
_respond = _prompt | _llm | StrOutputParser()

rag_chain = (
    RunnablePassthrough
    .assign(hits=RunnableLambda(_retrieve))          # adds  "hits"
    .assign(context=RunnableLambda(_build_context))  # adds  "context"
    .assign(response=_respond)                       # adds  "response" (the answer)
)


def answer(query: str, k: int = TOP_K, hybrid: bool = False) -> tuple[str, list[dict]]:
    """Run the full RAG pipeline once; return (answer_text, source_hits).

    Parameters
    ----------
    query  : the user's question
    k      : number of chunks to retrieve (Phase 1 only; Phase 2 uses config)
    hybrid : if True, use Phase 2 hybrid retrieval (BM25 + vector + reranking)
             instead of Phase 1 vector-only retrieval
    """
    if hybrid:
        # Phase 2: delegate retrieval to retriever.hybrid_retrieve(),
        # then reuse the same formatting + generation chain.
        from retriever import hybrid_retrieve
        hits    = hybrid_retrieve(query)
        context = _build_context({"hits": hits})
        response = _respond.invoke({"context": context, "question": query})
        return response, hits

    # Phase 1: original LCEL chain (unchanged)
    out = rag_chain.invoke({"question": query, "k": k})
    return out["response"], out["hits"]
