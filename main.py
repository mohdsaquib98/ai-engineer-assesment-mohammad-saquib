from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from groq import Groq

from config import settings
from dataset import DatasetRetriever
from groq_client import LLMError, answer_question
from models import AskRequest, AskResponse
from superhero import SuperheroAdapter

app = FastAPI(title="Superhero + Science Facts Chatbot", version="1.0.0")

dataset: DatasetRetriever | None = None
superhero_adapter: SuperheroAdapter | None = None
groq_client: Groq | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global dataset, superhero_adapter, groq_client
    dataset = DatasetRetriever(
        data_path=settings.DATASET_PATH,
        threshold=settings.DATASET_SEARCH_THRESHOLD,
    )
    superhero_adapter = SuperheroAdapter(
        token=settings.SUPERHERO_API_TOKEN,
        timeout=settings.SUPERHERO_API_TIMEOUT,
        max_failures=settings.SUPERHERO_API_MAX_FAILURES,
    )
    groq_client = Groq(api_key=settings.GROQ_API_KEY)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if dataset is None or superhero_adapter is None or groq_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still starting up. Please try again shortly.",
        )

    try:
        answer, tools_used = answer_question(
            client=groq_client,
            model=settings.GROQ_MODEL,
            question=request.question,
            dataset=dataset,
            superhero_adapter=superhero_adapter,
            dataset_top_k=settings.DATASET_TOP_K,
        )
        return AskResponse(answer=answer, tools_used=tools_used)
    except LLMError as e:
        print(f"LLM error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to process your question right now. Please try again.",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
