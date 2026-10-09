"""
Core logic: a single Groq-backed "agent" using native function-calling.

Flow:
  1. Send the question + tool schemas to Groq, let the model decide which
     tool(s) to call (dataset search, superhero search, or neither).
  2. Execute whichever tools were requested.
  3. Send the tool results back to Groq for a final, synthesized answer.

No framework - just the `groq` SDK (OpenAI-compatible chat completions API).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Tuple

from dataset import DatasetRetriever
from superhero import SuperheroAdapter

if TYPE_CHECKING:
    from groq import Groq

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about two topics only:

1. General science facts (space, biology, geology, physics) - use the search_dataset tool.
2. Superheroes / comic book characters - use the search_superhero tool.

Rules:
- Decide which tool(s) are relevant and call them. You may call both if the question spans
  both topics.
- If the question is about neither topic, do not call any tool - politely explain that you can
  only answer questions about science facts or superheroes.
- Only use information returned by the tools. Do not rely on prior knowledge for facts or
  superhero details.
- If a tool returns no relevant information, say so honestly rather than guessing.
- Always end your answer with a line stating which source(s) you used, e.g.
  "Source: local science facts dataset" or "Source: superheroapi.com" (or both)."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_dataset",
            "description": "Search a local dataset of general science facts (space, biology, geology, physics).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for the science facts dataset"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_superhero",
            "description": "Look up a superhero or comic book character by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the superhero to look up"}
                },
                "required": ["name"],
            },
        },
    },
]

MAX_QUESTION_LENGTH_FOR_LLM = 2000  # pre-LLM input safety check
MAX_TOOL_ARG_LENGTH = 200  # pre-tool argument validation
MAX_TOOL_RESULT_LENGTH = 2000  # post-tool output validation


class LLMError(Exception):
    """Raised when the Groq API call fails or returns something unusable."""


def _format_dataset_result(dataset: DatasetRetriever, query: str, top_k: int) -> str:
    results = dataset.search(query, top_k=top_k)
    if not results:
        return "No relevant facts found in the dataset for this query."
    return "\n".join(f"- {r.fact.title}: {r.fact.text}" for r in results)


def _format_superhero_result(adapter: SuperheroAdapter, name: str) -> str:
    result = adapter.search(name)
    if not result.found or not result.hero:
        return f"No superhero found matching '{name}'. ({result.error})"

    hero = result.hero
    bio = hero.get("biography", {})
    return (
        f"Name: {hero.get('name')}\n"
        f"Full name: {bio.get('full-name', 'N/A')}\n"
        f"Alignment: {bio.get('alignment', 'N/A')}\n"
        f"Power stats: {hero.get('powerstats', {})}"
    )


def _validate_tool_args(tool_name: str, args: dict) -> str:
    """Pre-tool validation. Returns the extracted primary arg string, raises LLMError if invalid."""
    arg_value = args.get("query") if tool_name == "search_dataset" else args.get("name")
    if not arg_value or not str(arg_value).strip():
        raise LLMError(f"Model called '{tool_name}' with an empty argument")
    if len(str(arg_value)) > MAX_TOOL_ARG_LENGTH:
        raise LLMError(f"Model called '{tool_name}' with an argument that is too long")
    return str(arg_value).strip()


def _validate_tool_output(tool_name: str, output: str) -> str:
    """Post-tool validation/sanitization. Caps output length before it goes back to the LLM."""
    if len(output) > MAX_TOOL_RESULT_LENGTH:
        output = output[:MAX_TOOL_RESULT_LENGTH] + "... (truncated)"
    return output


def answer_question(
    client: Groq,
    model: str,
    question: str,
    dataset: DatasetRetriever,
    superhero_adapter: SuperheroAdapter,
    dataset_top_k: int = 3,
) -> Tuple[str, List[str]]:
    """Returns (answer, tools_used). Raises LLMError on unrecoverable failure."""
    # pre-LLM input safety check
    if not question or not question.strip():
        raise LLMError("Question cannot be empty")
    if len(question) > MAX_QUESTION_LENGTH_FOR_LLM:
        raise LLMError("Question is too long")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        first_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
    except Exception as e:
        raise LLMError(f"Groq routing call failed: {e}") from e

    message = first_response.choices[0].message
    tool_calls = message.tool_calls or []

    if not tool_calls:
        # Model decided neither tool applies - its own text is the final answer.
        return message.content or (
            "I can only answer questions about superheroes or science facts."
        ), []

    messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in tool_calls
    ]})

    tools_used: List[str] = []
    tool_results: List[Tuple[str, str]] = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            raise LLMError(f"Model returned invalid arguments for '{tool_name}': {e}") from e

        arg_value = _validate_tool_args(tool_name, args)

        if tool_name == "search_dataset":
            result = _format_dataset_result(dataset, arg_value, dataset_top_k)
        elif tool_name == "search_superhero":
            result = _format_superhero_result(superhero_adapter, arg_value)
        else:
            result = f"Unknown tool: {tool_name}"

        result = _validate_tool_output(tool_name, result)
        tools_used.append(tool_name)
        tool_results.append((tool_name, result))

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # Enforce dataset retrieval for mixed superhero + science questions.
    # The LLM router can sometimes select only the superhero tool even when the
    # question also asks about a science concept (e.g. "Can Batman run at light speed?").
    lower_question = question.lower()
    science_cues = (
        "speed of light", "light speed", "light travel", "gravity", "photosynthesis",
        "atmosphere", "planet", "solar system", "black hole", "earth's rotation",
        "water", "dna", "volcano", "rainbow", "science",
    )
    has_science_cue = any(cue in lower_question for cue in science_cues)
    if "search_superhero" in tools_used and "search_dataset" not in tools_used and has_science_cue:
        dataset_result = _format_dataset_result(dataset, question, dataset_top_k)
        dataset_result = _validate_tool_output("search_dataset", dataset_result)
        tools_used.append("search_dataset")
        tool_results.append(("search_dataset", dataset_result))

    # Start a fresh text-only conversation for synthesis. Replaying the assistant's
    # native tool-call message can cause some models to emit another tool call here.
    source_by_tool = {
        "search_dataset": "local science facts dataset",
        "search_superhero": "superheroapi.com",
    }
    source_labels = list(dict.fromkeys(
        source_by_tool[name] for name, _ in tool_results if name in source_by_tool
    ))
    formatted_results = "\n\n".join(
        f"Tool: {name}\nResult:\n{result}" for name, result in tool_results
    )

    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the tool results provided. "
                "Do not call tools, invent facts, or use outside knowledge. "
                "If the results are insufficient, say so clearly. "
                f"End with exactly this source line: Source: {', '.join(source_labels)}"
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nTool results:\n{formatted_results}",
        },
    ]

    try:
        final_response = client.chat.completions.create(
            model=model,
            messages=synthesis_messages,
            temperature=0,
        )
    except Exception as e:
        raise LLMError(f"Groq synthesis call failed: {e}") from e

    answer = final_response.choices[0].message.content
    if not answer or not answer.strip():
        raise LLMError("Groq returned an empty final answer")

    return answer.strip(), list(dict.fromkeys(tools_used))
