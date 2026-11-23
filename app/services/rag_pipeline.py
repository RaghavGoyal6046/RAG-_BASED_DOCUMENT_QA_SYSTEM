import time
import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.models.schemas import QueryRequest, QueryResponse, SourceCitation
from app.services.vector_store import VectorStoreManager

RAG_PROMPT_TEMPLATE = """You are a precise, factual Document Question Answering assistant.
Answer the user's question STRICTLY based on the provided document context below.

Rules to strictly avoid hallucinations:
1. Ground your answer completely in the context provided below.
2. If the context does not contain enough information to answer the question, state:
   "Based on the provided documents, I could not find information to answer this question."
3. Do not make up facts or extrapolate beyond what is explicitly stated in the context snippets.
4. When stating a fact, refer to the document name and page/chunk where appropriate.

Context:
{context}

Question: {question}

Answer:"""

class RAGPipelineService:
    """
    RAG Pipeline orchestrator:
    - Queries vector store for relevant chunks
    - Performs context filtering & hallucination guards
    - Formats grounded prompt
    - Invokes LLM (Groq, Gemini, or Offline Extractive Generator)
    - Emits structured QueryResponse with source citations and confidence metrics
    """

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def _get_llm(self):
        """Initializes the active LLM based on settings and available API keys."""
        provider = settings.llm_provider.lower()

        if provider == "groq":
            if settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here":
                try:
                    from langchain_groq import ChatGroq
                    return ChatGroq(
                        model=settings.llm_model,
                        temperature=settings.llm_temperature,
                        api_key=settings.groq_api_key
                    )
                except Exception as e:
                    print(f"[Warning] Failed to instantiate ChatGroq ({e}). Falling back to offline generator.")
            else:
                print("[Info] Groq API key not configured. Using offline grounded generator.")

        elif provider == "gemini":
            if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
                try:
                    from google import genai
                    class GeminiClientWrapper:
                        def __init__(self, key: str, model: str):
                            self.client = genai.Client(api_key=key)
                            self.model = model
                        def invoke(self, prompt_text: str):
                            res = self.client.models.generate_content(model=self.model, contents=prompt_text)
                            return type("Obj", (), {"content": res.text})()

                    return GeminiClientWrapper(settings.gemini_api_key, settings.llm_model)
                except Exception as e:
                    print(f"[Warning] Failed to instantiate Gemini client ({e}). Falling back to offline generator.")
            else:
                print("[Info] Gemini API key not configured. Using offline grounded generator.")

        return None

    def _offline_grounded_answer(self, question: str, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Extractive grounded QA synthesizer used when running offline or without API keys.
        Extracts the most relevant sentences matching the question keywords from retrieved chunks.
        """
        if not retrieved_docs:
            return "Based on the provided documents, I could not find information to answer this question."

        q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
        stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "the", "and", "is", "are"}
        q_keywords = q_words - stop_words

        matched_sentences = []
        for doc, score in retrieved_docs:
            sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_words = set(re.findall(r"\b\w{3,}\b", s_clean.lower()))
                overlap = len(q_keywords.intersection(s_words))
                if overlap > 0:
                    matched_sentences.append((s_clean, overlap, score, doc.metadata.get("filename", "document")))

        if not matched_sentences:
            top_doc = retrieved_docs[0][0]
            first_sentence = top_doc.page_content.split("\n")[0]
            return f"According to {top_doc.metadata.get('filename', 'the document')}: {first_sentence}"

        matched_sentences.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_answers = []
        seen = set()
        for s, _, _, filename in matched_sentences[:3]:
            if s not in seen:
                seen.add(s)
                top_answers.append(s)

        doc_name = retrieved_docs[0][0].metadata.get("filename", "the indexed documents")
        return f"Based on {doc_name}:\n" + " ".join(top_answers)

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.time()
        k = request.top_k or settings.top_k_results
        threshold = request.similarity_threshold if request.similarity_threshold is not None else settings.similarity_threshold

        # 1. Retrieve top-k relevant document chunks
        retrieved_results = self.vector_store.similarity_search_with_score(
            query=request.question,
            k=k,
            threshold=threshold,
            document_ids=request.document_ids
        )

        # 2. Build citations
        sources: List[SourceCitation] = []
        for doc, score in retrieved_results:
            sources.append(SourceCitation(
                document_id=doc.metadata.get("document_id", "unknown"),
                document_name=doc.metadata.get("filename", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                page=doc.metadata.get("page"),
                content=doc.page_content,
                similarity_score=round(score, 4)
            ))

        # 3. Handle case where no relevant chunks met threshold
        if not retrieved_results:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            return QueryResponse(
                question=request.question,
                answer="Based on the provided documents, I could not find information to answer this question.",
                sources=[],
                confidence_score=0.0,
                has_grounded_context=False,
                latency_ms=latency_ms,
                model_used=settings.llm_model if self._get_llm() else "offline-grounded-qa"
            )

        # 4. Format context block for LLM prompt
        context_blocks = []
        for doc, score in retrieved_results:
            fname = doc.metadata.get("filename", "Document")
            page = doc.metadata.get("page", 1)
            chunk_idx = doc.metadata.get("chunk_index", 0)
            context_blocks.append(f"--- [Source: {fname} | Page: {page} | Chunk: {chunk_idx}] ---\n{doc.page_content}")

        context_str = "\n\n".join(context_blocks)
        avg_confidence = round(sum(score for _, score in retrieved_results) / len(retrieved_results), 4)

        # 5. Generate Answer via LLM or Grounded Extractor
        llm = self._get_llm()
        if llm:
            try:
                formatted_prompt = self.prompt.format(context=context_str, question=request.question)
                response = llm.invoke(formatted_prompt)
                answer_text = response.content if hasattr(response, "content") else str(response)
                model_name = settings.llm_model
            except Exception as e:
                print(f"[Warning] LLM invocation failed ({e}). Falling back to offline synthesizer.")
                answer_text = self._offline_grounded_answer(request.question, retrieved_results)
                model_name = "offline-fallback"
        else:
            answer_text = self._offline_grounded_answer(request.question, retrieved_results)
            model_name = "offline-grounded-qa"

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return QueryResponse(
            question=request.question,
            answer=answer_text.strip(),
            sources=sources,
            confidence_score=avg_confidence,
            has_grounded_context=True,
            latency_ms=latency_ms,
            model_used=model_name
        )
