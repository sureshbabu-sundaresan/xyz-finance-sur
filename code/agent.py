import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator, ValidationError

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# Constants for agent operation
SYSTEM_PROMPT = (
    "You are a professional assistant specializing in answering queries about the finance policies of XYZ organization. "
    "Your primary responsibility is to provide accurate, concise, and policy-compliant answers strictly based on the content retrieved from the official finance policy documentation (XYZ_Finance_Policies_Detailed.pdf) using Azure AI Search.\n\n"
    "Instructions:\n"
    "- Only use information present in the retrieved context from the authorized document.\n"
    "- Do not generate or infer answers beyond what is explicitly stated in the source.\n"
    "- If the answer is not found in the retrieved context, politely inform the user that the information is not available.\n"
    "- Maintain a formal and professional tone in all responses.\n"
    "- Do not answer queries outside the scope of XYZ finance policies.\n\n"
    "Output Format:\n"
    "- Provide clear, direct answers to the user's question.\n"
    "- Reference the relevant policy section if applicable.\n"
    "- If information is not available, use the fallback response.\n\n"
    "Fallback Behavior:\n"
    "- If the answer cannot be found in the provided context, respond with the fallback message."
)
OUTPUT_FORMAT = (
    "- Direct, factual answer to the user's query\n"
    "- Reference to the relevant policy section if applicable\n"
    "- Professional and concise language\n"
    "- Use fallback response if information is not available"
)
FALLBACK_RESPONSE = "I'm sorry, I could not find information related to your query in the official XYZ finance policy documentation."

SELECTED_DOCUMENT_TITLES = ["XYZ_Finance_Policies_Detailed.pdf"]

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

class QueryRequest(BaseModel):
    user_query: str = Field(..., description="User query regarding the finance policy of XYZ organization")

    @model_validator(mode="after")
    def validate_content(cls, values):
        user_query = values.get("user_query", "")
        if not user_query or not isinstance(user_query, str):
            raise ValidationError("user_query must be a non-empty string")
        if len(user_query.strip()) == 0:
            raise ValidationError("user_query cannot be empty or whitespace")
        if len(user_query) > 50000:
            raise ValidationError("user_query exceeds maximum allowed length (50,000 characters)")
        values["user_query"] = user_query.strip()
        return values

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the query was processed successfully")
    answer: Optional[str] = Field(None, description="Agent's answer to the user query")
    error: Optional[str] = Field(None, description="Error message if applicable")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input errors")

class AuditLogger:
    """Logs all queries, responses, errors, and declined answers for compliance and monitoring."""
    def __init__(self):
        self.logger = logging.getLogger("agent.audit")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: dict) -> None:
        try:
            self.logger.info(f"EventType: {event_type} | Details: {json.dumps(details, default=str)}")
        except Exception as e:
            self.logger.warning(f"Audit log failed: {e}")

class ErrorHandler:
    """Handles errors, implements retry logic, fallback behaviors, and returns user-friendly error messages."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def handle_error(self, error: Exception, context: dict) -> str:
        error_msg = str(error)
        self.audit_logger.log_event("error", {"error": error_msg, "context": context})
        # Map known errors to business error codes/messages
        if isinstance(error, ValidationError):
            return "Invalid input: Please check your query and try again."
        elif "NO_CONTEXT_FOUND" in error_msg:
            return FALLBACK_RESPONSE
        elif "INVALID_QUERY" in error_msg:
            return "Your query is outside the scope of XYZ finance policies. Please ask about finance policy topics."
        else:
            return f"An unexpected error occurred: {error_msg}"

class ChunkRetriever:
    """Queries Azure AI Search using vector + keyword search, filters by selected_document_titles, returns top_k relevant chunks."""
    def __init__(self):
        self.search_client = None

    def _get_search_client(self):
        if self.search_client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            api_key = Config.AZURE_SEARCH_API_KEY
            if not endpoint or not index_name or not api_key:
                raise RuntimeError("Azure AI Search credentials not configured")
            self.search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self.search_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, document_titles: List[str], top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks from Azure AI Search using vector + keyword search, filtered by document titles."""
        search_client = self._get_search_client()
        openai_client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        _t0 = _time.time()
        embedding_resp = await openai_client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(embedding_resp)[:200] if embedding_resp is not None else None,
                status="success",
            )
        except Exception:
            pass

        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=top_k,
            fields="vector"
        )
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": ["chunk", "title"],
        }
        if document_titles:
            odata_parts = [f"title eq '{t}'" for t in document_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)
        _t1 = _time.time()
        results = search_client.search(**search_kwargs)
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t1) * 1000),
                output=str(results)[:200] if results is not None else None,
                status="success",
            )
        except Exception:
            pass
        context_chunks = [r["chunk"] for r in results if r.get("chunk")]
        return context_chunks

class LLMService:
    """Formats prompt (using enhanced system prompt), passes user query and retrieved chunks to Azure OpenAI GPT-4.1, returns generated response."""
    def __init__(self):
        self.openai_client = None

    def _get_llm_client(self):
        if self.openai_client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise RuntimeError("AZURE_OPENAI_API_KEY not configured")
            self.openai_client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self.openai_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def generate_response(self, prompt: str, context_chunks: List[str], user_query: str) -> str:
        """Calls Azure OpenAI GPT-4.1 with enhanced system prompt, user query, and retrieved chunks; returns LLM-generated answer."""
        messages = [
            {"role": "system", "content": prompt + "\n\nOutput Format: " + OUTPUT_FORMAT},
            {"role": "user", "content": f"{user_query}\n\nContext:\n" + "\n\n".join(context_chunks)}
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        response = await self._get_llm_client().chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4.1",
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4.1",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return sanitize_llm_output(content, content_type="text")

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

class FinancePolicyQueryAgent:
    """Coordinates the end-to-end flow: receives user query, invokes chunk retrieval, calls LLM, formats and returns response."""
    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.audit_logger = AuditLogger()
        self.error_handler = ErrorHandler(self.audit_logger)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_query(self, user_query: str) -> dict:
        """Orchestrates the query answering process: validates scope, retrieves chunks, calls LLM, handles errors, returns formatted response."""
        async with trace_step(
            "validate_input", step_type="parse",
            decision_summary="Validate user query for finance policy scope",
            output_fn=lambda r: f"valid={r}",
        ) as step:
            if not user_query or not isinstance(user_query, str) or len(user_query.strip()) == 0:
                error_msg = "INVALID_QUERY"
                self.audit_logger.log_event("declined_answer", {"reason": error_msg, "user_query": user_query})
                step.capture({"success": False, "error": error_msg})
                return {
                    "success": False,
                    "answer": None,
                    "error": self.error_handler.handle_error(Exception(error_msg), {"user_query": user_query}),
                    "tips": "Please provide a valid query about the finance policy of XYZ organization."
                }
            step.capture({"success": True})

        async with trace_step(
            "retrieve_chunks", step_type="tool_call",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks_found={len(r)}",
        ) as step:
            try:
                context_chunks = await self.chunk_retriever.retrieve_chunks(
                    query=user_query,
                    document_titles=SELECTED_DOCUMENT_TITLES,
                    top_k=5
                )
                step.capture({"chunks_found": len(context_chunks)})
            except Exception as e:
                self.audit_logger.log_event("error", {"stage": "retrieval", "error": str(e), "user_query": user_query})
                step.capture({"success": False, "error": str(e)})
                return {
                    "success": False,
                    "answer": None,
                    "error": self.error_handler.handle_error(e, {"user_query": user_query}),
                    "tips": "Please try again later or contact support if the issue persists."
                }

        if not context_chunks or len(context_chunks) == 0:
            self.audit_logger.log_event("declined_answer", {"reason": "NO_CONTEXT_FOUND", "user_query": user_query})
            return {
                "success": True,
                "answer": FALLBACK_RESPONSE,
                "error": None,
                "tips": None
            }

        async with trace_step(
            "generate_response", step_type="llm_call",
            decision_summary="Generate answer using LLM with retrieved context",
            output_fn=lambda r: f"answer={r[:100]}",
        ) as step:
            try:
                answer = await self.llm_service.generate_response(
                    prompt=SYSTEM_PROMPT,
                    context_chunks=context_chunks,
                    user_query=user_query
                )
                step.capture({"answer": answer[:100]})
            except Exception as e:
                self.audit_logger.log_event("error", {"stage": "llm", "error": str(e), "user_query": user_query})
                step.capture({"success": False, "error": str(e)})
                return {
                    "success": False,
                    "answer": None,
                    "error": self.error_handler.handle_error(e, {"user_query": user_query}),
                    "tips": "Please try again later or contact support if the issue persists."
                }

        self.audit_logger.log_event("query_response", {
            "user_query": user_query,
            "answer": answer,
            "context_chunks_count": len(context_chunks)
        })
        return {
            "success": True,
            "answer": answer,
            "error": None,
            "tips": None
        }

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="XYZ Finance Policy Query Agent",
    description="Answers queries about the finance policies of XYZ organization using Azure AI Search and Azure OpenAI GPT-4.1. Strictly policy-compliant, professional, and content-agnostic.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    """Main endpoint for answering finance policy queries."""
    agent = FinancePolicyQueryAgent()
    try:
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Parse and validate user input",
            output_fn=lambda r: f"user_query={r.get('user_query','')[:50]}",
        ) as step:
            user_query = req.user_query
            step.capture({"user_query": user_query[:50]})

        result = await agent.answer_query(user_query=user_query)
        answer = result.get("answer")
        error = result.get("error")
        tips = result.get("tips")
        return QueryResponse(
            success=result.get("success", False),
            answer=answer,
            error=error,
            tips=tips
        )
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "answer": None,
                "error": "Malformed JSON or invalid input. Please check your query and try again.",
                "tips": "Ensure your JSON is properly formatted and your query is not empty."
            }
        )
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "answer": None,
                "error": f"Internal server error: {str(e)}",
                "tips": "Please try again later or contact support if the issue persists."
            }
        )

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())