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


# ---------------------------------------------------------------------------
# CONVERSATION MEMORY (ROLLING SUMMARIZATION)
# ---------------------------------------------------------------------------

# Prompt used to compress turns that fall outside the recent window into an
# updated running summary. Instructs the LLM to keep specific facts (drugs,
# sections, error codes) because those are what later follow-ups reference.
_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You maintain a running summary of a conversation. You receive the "
         "current summary (possibly empty) plus some older conversation turns "
         "that are about to be discarded. Produce an updated, concise summary "
         "of 2-4 sentences that merges the existing summary with these turns. "
         "Crucially, PRESERVE specific facts and topics rather than vague "
         "generalities: exact names, drugs, sections, error codes, model "
         "numbers, numbers and anything concrete the user or assistant "
         "mentioned — these are what later follow-up questions will reference."),
        ("human",
         "Current summary:\n{existing_summary}\n\n"
         "Older turns being added:\n{turns}\n\n"
         "Write the updated summary ONLY (2-4 sentences, no preamble)."),
    ]
)

# Prompt used to rewrite a possibly-context-dependent question into a standalone
# one, so it can be fed to retrieval on its own.
_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You rewrite the user's latest question into a standalone question. "
         "Use the conversation summary and recent turns as context to resolve "
         "pronouns and implicit references. Include all necessary context so "
         "the rewritten question is self-contained. "
         "If the question is already standalone, return it UNCHANGED. "
         "Output only the rewritten question."),
        ("human",
         "Conversation summary (if any):\n{summary}\n\n"
         "Recent turns:\n{recent}\n\n"
         "Latest question: {question}\n\n"
         "Standalone question:"),
    ]
)


def _format_turns(turns: list) -> str:
    """Format a list of message dicts as 'role: content' lines."""
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def _update_summary(existing_summary: str, old_turns: list) -> str:
    """Compress old_turns into an updated running summary using _llm."""
    if not old_turns and not existing_summary:
        return ""
    turns = _format_turns(old_turns)
    resp  = _llm.invoke(
        _SUMMARY_PROMPT.format_messages(
            existing_summary=existing_summary or "(no summary yet)",
            turns=turns or "(no turns)",
        )
    )
    return resp.content.strip()


# NOTE: This trades one extra LLM call — but only when the window actually
# needs to shift, NOT every turn — for preserving long-range conversational
# context that a plain sliding window would silently discard. Old turns are
# compressed into a running summary instead of being dropped.
def _manage_history(session_state: dict, max_recent_turns: int = 6) -> dict:
    """Keep recent_turns at most max_recent_turns by summarizing the overflow.

    session_state shape: {"summary": str, "recent_turns": [{"role", "content"}, ...]}
    Moves the oldest excess turns out, folds them into the running summary via
    _update_summary(), and trims recent_turns. Returns the updated state.
    """
    recent = session_state.get("recent_turns", [])
    if len(recent) <= max_recent_turns:
        return session_state  # nothing to trim yet

    excess   = recent[: len(recent) - max_recent_turns]
    remaining = recent[len(recent) - max_recent_turns:]
    new_summary = _update_summary(session_state.get("summary", ""), excess)

    return {
        "summary": new_summary,
        "recent_turns": remaining,
    }


def _condense_query(session_state: dict, current_question: str) -> str:
    """Rewrite current_question into a standalone question using memory."""
    summary = session_state.get("summary", "")
    recent  = session_state.get("recent_turns", [])

    if not summary and not recent:
        return current_question  # very first turn — nothing to attach

    recent_str = _format_turns(recent) or "(no recent turns)"
    resp = _llm.invoke(
        _CONDENSE_PROMPT.format_messages(
            summary=summary or "(no summary)",
            recent=recent_str,
            question=current_question,
        )
    )
    condensed = resp.content.strip()
    return condensed or current_question


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


def answer(
    query: str,
    k: int = TOP_K,
    hybrid: bool = False,
    session_state: dict | None = None,
) -> tuple[str, list[dict], dict]:
    """Run the full RAG pipeline once; return (answer_text, source_hits, session_state).

    Parameters
    ----------
    query          : the user's question
    k              : number of chunks to retrieve (Phase 1 only; Phase 2 uses config)
    hybrid         : if True, use Phase 2 hybrid retrieval (BM25 + vector + reranking)
                     instead of Phase 1 vector-only retrieval
    session_state  : conversation memory {"summary": str, "recent_turns": list}.
                     Defaults to an empty memory dict if not provided. Mutated and
                     returned so the caller can persist it across turns.

    Memory flow
    -----------
    1. _manage_history() trims recent_turns beyond the window, folding overflow
       into the running summary (extra LLM call only when the window shifts).
    2. _condense_query() builds a standalone query from memory + current question.
       Retrieval uses THIS condensed query.
    3. Generation uses the ORIGINAL raw question, plus summary + recent_turns,
       so the final answer's phrasing stays natural.
    """
    if session_state is None:
        session_state = {"summary": "", "recent_turns": []}

    # 1. Update/trim memory first (may trigger summarization).
    session_state = _manage_history(session_state)

    # 2. Build a standalone query for retrieval from the updated memory.
    retrieval_query = _condense_query(session_state, query)

    # Build a context-block prefix carrying the conversation memory (if any).
    memory_prefix = ""
    if session_state.get("summary"):
        memory_prefix += f"Conversation summary:\n{session_state['summary']}\n\n"
    if session_state.get("recent_turns"):
        memory_prefix += "Recent conversation:\n" + _format_turns(
            session_state["recent_turns"]) + "\n\n"

    if hybrid:
        # Phase 2: delegate retrieval to retriever.hybrid_retrieve(),
        # then reuse the same formatting + generation chain.
        from retriever import hybrid_retrieve
        hits    = hybrid_retrieve(retrieval_query)
        context = _build_context({"hits": hits})
        # Final generation sees memory + original raw question.
        full_ctx = memory_prefix + context
        response = _respond.invoke({"context": full_ctx, "question": query})
        return response, hits, session_state

    # Phase 1: original LCEL chain, but with a memory-augmented prompt.
    p1_hits = _retrieve({"question": retrieval_query, "k": k})
    vs_ctx = _build_context({"hits": p1_hits})
    full_ctx = memory_prefix + vs_ctx
    response = _respond.invoke({"context": full_ctx, "question": query})
    return response, p1_hits, session_state
