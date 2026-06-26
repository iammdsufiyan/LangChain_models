"""
Advanced Prompt Engineering Engine
====================================
A 6-stage production pipeline stacking the most powerful prompt engineering
techniques on top of LangChain LCEL + Groq (llama-3.3-70b-versatile).

  Stage 1  ──  Meta-Prompting          LLM generates its own optimal system prompt
  Stage 2  ──  Chain-of-Thought        Forced structured step-by-step reasoning
  Stage 3  ──  Tree of Thoughts        3 parallel reasoning branches (RunnableParallel)
  Stage 4  ──  Self-Critique Loop      Model synthesizes & scores; auto-rewrites if < 8
  Stage 5  ──  Constitutional AI       Principle-based revision pass
  Stage 6  ──  Structured Extraction   Typed Pydantic output via PydanticOutputParser

  Bonus    ──  Few-Shot Examples       Static examples injected as system messages
  Bonus    ──  Streaming Chat          Real-time token output with full message history
  Bonus    ──  LCEL Chains             All stages composed with | operator

HOW THE STAGES CONNECT (data flow):
─────────────────────────────────────────────────────────────────────
  User Query
      │
      ▼  meta_chain
  [Stage 1] system_prompt  ←── LLM writes the best prompt for your query
      │
      ▼  cot_chain  (receives: system_prompt + query)
  [Stage 2] cot_reasoning  ←── 5-step labeled thinking
      │
      ▼  tot_parallel  (receives: query + cot_reasoning)
  [Stage 3] branches dict  ←── 3 parallel LLM calls at the same time
      │  analytical, creative, critical
      ▼  critique_chain  (receives: query + all 3 branches)
  [Stage 4] critique dict  ←── synthesis + score + needs_revision flag
      │
      ├─ if needs_revision == True
      │       ▼  revision_chain
      │   [Stage 5] final_answer (revised, principled)
      │
      └─ if needs_revision == False
              final_answer = critique["synthesis"]  (no revision needed)
      │
      ▼  extraction_chain  (receives: query + final_answer)
  [Stage 6] AnalysisResult  ←── typed Python object with all fields
─────────────────────────────────────────────────────────────────────
"""

# ── future annotations: allows type hints to reference classes defined later
from __future__ import annotations

import textwrap          # for cleaning up multi-line strings (removes leading spaces)
from typing import Any, List  # type hints — List[str] means a list of strings

from dotenv import load_dotenv   # reads your .env file so GROQ_API_KEY is available
from pydantic import BaseModel, Field  # for defining the typed output schema in Stage 6

# LangChain message types — each represents a role in the conversation:
#   SystemMessage  = instructions to the AI (the "rules" message)
#   HumanMessage   = what the user says
#   AIMessage      = what the AI previously said (used in chat history)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Output parsers — convert the model's raw text into usable Python objects:
#   StrOutputParser    = returns a plain string
#   JsonOutputParser   = parses model output as JSON → Python dict
#   PydanticOutputParser = parses JSON → typed Pydantic object (Stage 6)
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser

# ChatPromptTemplate — builds a structured prompt from a list of messages
# Each message is a tuple like ("system", "text") or ("human", "{variable}")
# Variables inside {} are filled in when you call .invoke({"variable": value})
from langchain_core.prompts import ChatPromptTemplate

# RunnableParallel — runs multiple chains at the exact same time (Stage 3)
# Instead of running analytical → creative → critical one by one,
# RunnableParallel fires all three simultaneously, saving time
from langchain_core.runnables import RunnableParallel

# ChatGroq — the LangChain connector for the Groq API
# Groq runs LLaMA models at very high speed using their LPU hardware
from langchain_groq import ChatGroq

# Load .env file so GROQ_API_KEY is available to ChatGroq
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INSTANCES
# We create THREE separate model objects, each with a different temperature.
# Temperature controls how "random" or "creative" the model is:
#   temperature=0.1  → very focused, deterministic, prefers the most likely token
#   temperature=0.4  → slightly varied, good for balanced tasks
#   temperature=0.75 → more creative, explores less obvious ideas
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ID = "llama-3.3-70b-versatile"  # the specific Groq model to use

# Used in: Stage 2 (CoT), Stage 3 analytical branch, Stage 3 critical branch,
#          Stage 4 (self-critique), Stage 6 (structured extraction)
# Why low temp? These stages need accurate, logical, consistent answers.
model_precise  = ChatGroq(model=MODEL_ID, temperature=0.1,  max_completion_tokens=2048)

# Used in: Stage 3 creative branch, Stage 5 (constitutional revision),
#          Streaming chat
# Why higher temp? Creative thinking and revision benefit from more variety.
model_creative = ChatGroq(model=MODEL_ID, temperature=0.75, max_completion_tokens=2048)

# Used in: Stage 1 (meta-prompt generation)
# Why medium temp? Meta-prompting needs some creativity but not too much randomness.
model_meta     = ChatGroq(model=MODEL_ID, temperature=0.4,  max_completion_tokens=600)

# StrOutputParser: the simplest parser — just returns the model's reply as a string.
# Used at the end of chains where we want plain text output.
str_parser = StrOutputParser()


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMA  —  AnalysisResult
#
# This class defines what the FINAL output of the entire pipeline looks like.
# Pydantic validates that all fields exist and have the correct types.
# Stage 6 uses PydanticOutputParser to force the model to produce this exact shape.
#
# Relationship: This class is the return type of run_pipeline().
#               display_result() reads from this class to print the output.
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisResult(BaseModel):
    # One-paragraph executive summary of the entire answer
    summary: str = Field(description="One-paragraph executive summary")

    # A list of the most important takeaways (3-5 bullet points)
    key_insights: List[str] = Field(description="3-5 high-value insights")

    # How confident the model is (0.0 = no idea, 1.0 = completely certain)
    # ge=0.0, le=1.0 means Pydantic will reject values outside this range
    confidence_score: float = Field(ge=0.0, le=1.0, description="Model confidence 0–1")

    # Model's self-rating of its own reasoning quality
    # ge=1, le=10 means must be between 1 and 10 inclusive
    reasoning_quality: int = Field(ge=1, le=10, description="Self-rated quality 1–10")

    # What this answer doesn't cover or might be wrong about
    limitations: List[str] = Field(description="Known limitations or gaps in this answer")

    # Questions the user should ask next to go deeper
    follow_up_questions: List[str] = Field(description="3 powerful follow-up questions")


# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES  (injected into Stage 2)
#
# Few-shot prompting = showing the model 2-3 examples of ideal answers
# BEFORE asking it the actual question. This "teaches by example."
#
# Why as a plain string (not FewShotChatMessagePromptTemplate)?
# Embedding FewShotChatMessagePromptTemplate inside ChatPromptTemplate.from_messages()
# can break variable resolution ({query}) in some LangChain versions.
# A plain system message string is simpler and works across all versions.
#
# Relationship: _FEW_SHOT_EXAMPLES is used inside _cot_template (Stage 2)
#               as the second system message, right before the CoT instructions.
# ══════════════════════════════════════════════════════════════════════════════

_FEW_SHOT_EXAMPLES = textwrap.dedent("""\
    Here are two examples of the high-quality, structured responses expected:

    EXAMPLE 1
    Q: Explain transformers in AI
    A: Core Mechanism: Transformers use self-attention to weigh every token against every
    other — O(n²) but fully parallelizable across GPUs.
    Key Innovation: Multi-head attention lets the model attend to different representation
    subspaces simultaneously, capturing syntax, semantics, and coreference at once.
    Why It Matters: Replaced sequential RNNs entirely; scaling laws showed that larger
    transformers trained on more data consistently improve — leading to GPT-4, Gemini, Claude.

    EXAMPLE 2
    Q: What is prompt engineering?
    A: Definition: The discipline of crafting LLM inputs that reliably elicit desired
    outputs — bridging human intent and model behavior without touching model weights.
    Key Techniques: Zero-shot, few-shot, chain-of-thought, tree-of-thoughts,
    self-consistency, meta-prompting, constitutional AI, retrieval augmentation.
    Mental Model: Programming in natural language — precision, examples, and explicit
    constraints are your type annotations and compiler flags.
""")
# textwrap.dedent() removes the common leading whitespace from all lines,
# so the string looks clean without extra indentation.


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — META-PROMPT GENERATOR
#
# WHAT IT DOES:
#   Takes the user's raw query and asks the LLM to write the BEST possible
#   system prompt for that query. The LLM essentially programs itself.
#
# HOW IT WORKS:
#   _meta_template → model_meta → str_parser
#   The | operator chains these together into meta_chain.
#   When you call meta_chain.invoke({"query": "..."}), LangChain:
#     1. Fills {query} in the human message
#     2. Sends the filled prompt to model_meta
#     3. Passes the reply through str_parser to get a plain string
#
# OUTPUT: a system prompt string like:
#   "You are an expert in distributed systems with 20 years of experience..."
#
# RELATIONSHIP:
#   → Output (system_prompt) is fed into Stage 2 as the first system message
# ══════════════════════════════════════════════════════════════════════════════

# _meta_template defines the 2-message conversation sent to the model:
#   Message 1 (system): explains the meta-prompting task to the model
#   Message 2 (human): contains the actual user query inside {query}
_meta_template = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""\
        You are a world-class prompt engineer with deep expertise in LLM behavior.
        Given a user query, generate the single most effective system prompt that will
        make an LLM answer it with maximum accuracy, depth, and clarity.

        Output ONLY the system prompt text — no preamble, no quotes, no explanation.

        The system prompt must:
        - Define a precise expert persona relevant to the query's domain
        - Set the optimal reasoning style (analytical / creative / critical / hybrid)
        - Specify output format constraints (structure, length, tone, depth)
        - Inject domain-specific heuristics, mental models, or cognitive frameworks
        - Instruct the model to think before answering (reasoning-first)
    """)),
    # {query} is a LangChain template variable — replaced with the real query at runtime
    ("human", "User query: {query}\n\nOptimal system prompt:"),
])

# meta_chain = the full pipeline for Stage 1
# _meta_template formats the messages → model_meta generates the system prompt →
# str_parser converts the model's response object into a plain Python string
meta_chain = _meta_template | model_meta | str_parser


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CHAIN-OF-THOUGHT REASONER
#
# WHAT IT DOES:
#   Forces the model to reason step by step through the problem, labeling each
#   stage of its thinking explicitly. This reduces errors because the model
#   must justify each conclusion before moving to the next one.
#
# HOW IT WORKS:
#   _cot_template has 4 messages in order:
#     1. system = the custom system prompt generated by Stage 1
#     2. system = the few-shot examples (shows what good answers look like)
#     3. system = the Chain-of-Thought instructions (the 5 steps)
#     4. human  = the actual user query
#
#   The model sees all 4 messages together and produces a structured response
#   with [STEP 1 — UNDERSTAND], [STEP 2 — DECOMPOSE], etc.
#
# INPUT:  {"system_prompt": <from Stage 1>, "query": <user's question>}
# OUTPUT: a long string with all 5 CoT steps written out
#
# RELATIONSHIP:
#   → Receives system_prompt from Stage 1
#   → Output (cot_reasoning) is passed to all 3 branches in Stage 3
# ══════════════════════════════════════════════════════════════════════════════

_cot_template = ChatPromptTemplate.from_messages([
    # {system_prompt} is filled with Stage 1's output — the model's self-generated instructions
    ("system", "{system_prompt}"),

    # _FEW_SHOT_EXAMPLES is a plain string — the 2 example Q&A pairs
    # This tells the model "your answer should look like these examples"
    ("system", _FEW_SHOT_EXAMPLES),

    # These are the mandatory CoT instructions — the model MUST follow this structure
    # The 5 labeled steps ensure the model doesn't skip ahead without reasoning
    ("system", textwrap.dedent("""\
        You MUST reason using this explicit Chain-of-Thought structure.
        Label every step clearly in your response:

        [STEP 1 — UNDERSTAND]   Restate the core question in your own precise words.
        [STEP 2 — DECOMPOSE]    Break it into 3–5 concrete sub-problems.
        [STEP 3 — ANALYZE]      Solve each sub-problem with evidence, logic, and examples.
        [STEP 4 — SYNTHESIZE]   Combine all sub-answers into one coherent response.
        [STEP 5 — VERIFY]       Check for contradictions, unsupported claims, or gaps.
                                State what you are uncertain about using [UNCERTAIN: ...].
    """)),

    # {query} is the user's actual question — filled in at invoke() time
    ("human", "{query}"),
])

# cot_chain = Stage 2 full pipeline
# _cot_template formats messages → model_precise reasons carefully →
# str_parser returns the full 5-step reasoning as a string
cot_chain = _cot_template | model_precise | str_parser


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — TREE OF THOUGHTS (3 PARALLEL BRANCHES)
#
# WHAT IT DOES:
#   Runs 3 completely independent "thinkers" on the same problem simultaneously.
#   Each thinker has a different personality and reasoning style:
#     - Analytical: data-driven, logical, precise
#     - Creative:   lateral, cross-domain, surprising
#     - Critical:   skeptical, finds flaws, stress-tests
#
# HOW IT WORKS:
#   _make_branch() is a FACTORY FUNCTION — it creates a chain with a given persona.
#   RunnableParallel fires all 3 chains at the same time (not one by one).
#   It returns a dict: {"analytical": "...", "creative": "...", "critical": "..."}
#
# INPUT:  {"query": <user question>, "cot_reasoning": <from Stage 2>}
# OUTPUT: a dict with 3 keys, each containing one branch's full answer
#
# RELATIONSHIP:
#   → Receives query (original) + cot_reasoning (from Stage 2)
#   → All 3 branch outputs are passed together to Stage 4 for synthesis
# ══════════════════════════════════════════════════════════════════════════════

def _make_branch(persona: str, instruction: str, model: ChatGroq):
    """
    Factory function that creates one Tree-of-Thoughts branch.

    How it works:
      - Takes a persona string (who the AI is), an instruction string (how to think),
        and a model instance (which temperature/model to use).
      - Builds a ChatPromptTemplate with 2 messages:
          1. system = persona + instruction (defines the thinker's identity)
          2. human  = the query + previous CoT reasoning to build on
      - Returns a complete LCEL chain: template | model | str_parser

    Why a factory function?
      Because all 3 branches have the same STRUCTURE but different content.
      Instead of writing the same template 3 times, _make_branch() takes
      the varying parts as arguments. This is the DRY principle (Don't Repeat Yourself).

    Returns: an LCEL chain (RunnableSequence) ready to be called with .invoke()
    """
    # Build the prompt — same structure for all branches, different persona+instruction
    template = ChatPromptTemplate.from_messages([
        # f-string combines persona and instruction into one system message
        ("system", f"{persona}\n\n{instruction}"),

        # Both {query} and {cot_reasoning} must be passed when this chain is invoked
        ("human", "Query: {query}\n\nExisting reasoning to build on:\n{cot_reasoning}"),
    ])
    # Return the composed chain — | chains these 3 objects into a pipeline
    return template | model | str_parser


# BRANCH 1 — Analytical
# Uses model_precise (temp=0.1) because analytical thinking needs consistency
# Persona: scientist/statistician — focused on facts, logic, measurable claims
analytical_branch = _make_branch(
    "You are a rigorous analytical thinker — a scientist, statistician, and logician.",
    "Focus exclusively on: data, formal logic, causality, measurable evidence, definitions. "
    "Identify what can be proven versus what is speculation. Challenge every assumption. "
    "Prefer precision over breadth. Quantify wherever possible.",
    model_precise,  # low temperature → deterministic, factual answers
)

# BRANCH 2 — Creative
# Uses model_creative (temp=0.75) because creative thinking needs variability
# Persona: innovator/polymath — finds unexpected angles and analogies
creative_branch = _make_branch(
    "You are a lateral, creative thinker — an innovator, systems architect, and polymath.",
    "Explore: unexpected angles, cross-domain analogies, second-order effects, emergent patterns. "
    "Ask 'what if this were completely different?', 'what else could this mean?', "
    "'what would a biologist / economist / artist say?' Surprise the reader.",
    model_creative,  # higher temperature → more varied, creative responses
)

# BRANCH 3 — Critical
# Uses model_precise (temp=0.1) because finding flaws needs careful, focused reasoning
# Persona: skeptic/devil's advocate — stress-tests every assumption
critical_branch = _make_branch(
    "You are a hardcore skeptic and stress-tester — a devil's advocate and adversarial reviewer.",
    "Your job: find weaknesses, counterarguments, hidden assumptions, edge cases, failure modes. "
    "What would the harshest critic say? Where does the reasoning break down? "
    "What is being glossed over? What would falsify this? Be relentless but constructive.",
    model_precise,  # low temperature → systematic, thorough critique
)

# RunnableParallel wraps all 3 branches into a single runnable object.
# When tot_parallel.invoke(inputs) is called:
#   - It sends the SAME inputs to all 3 branches simultaneously
#   - All 3 LLM calls happen at the same time (parallel API requests)
#   - It waits for all 3 to finish
#   - Returns: {"analytical": "...", "creative": "...", "critical": "..."}
# This is much faster than running them one by one (3x speedup approximately).
tot_parallel = RunnableParallel(
    analytical=analytical_branch,  # key name "analytical" → branch output stored here
    creative=creative_branch,       # key name "creative"   → branch output stored here
    critical=critical_branch,       # key name "critical"   → branch output stored here
)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SELF-CRITIQUE & SYNTHESIS
#
# WHAT IT DOES:
#   Reads all 3 branch outputs and combines them into ONE superior answer.
#   Then the model scores its own synthesis on a 1-10 scale and lists weaknesses.
#   If score < 8, it sets needs_revision=True, which triggers Stage 5.
#
# HOW IT WORKS:
#   The model receives all 3 branches in a single large prompt and is asked
#   to return a JSON object (not free text). JsonOutputParser converts this JSON
#   string into a Python dict automatically.
#
# INPUT:  {"query": ..., "analytical": ..., "creative": ..., "critical": ...}
#         (the **branches dict unpacking passes the 3 branch outputs)
# OUTPUT: Python dict like:
#         {
#           "synthesis": "the best combined answer",
#           "score": 7,
#           "needs_revision": True,
#           "weaknesses": ["missing examples", "too abstract"]
#         }
#
# RELATIONSHIP:
#   → Receives all 3 branch outputs from Stage 3
#   → "synthesis" and "score" determine whether Stage 5 runs
#   → "synthesis" becomes final_answer if score >= 8
# ══════════════════════════════════════════════════════════════════════════════

# Note: {{ and }} in the system message are ESCAPED curly braces.
# In Python f-strings and LangChain templates, {var} means "insert variable here".
# To literally print { or } in the output, you must write {{ or }}.
# The JSON example in the system prompt uses {{ and }} so they appear as { and }.
_critique_template = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""\
        You are an expert evaluator, synthesizer, and quality judge.
        You have received three independent analyses of the same query from three
        different thinking personas: analytical, creative, and critical.

        Your tasks:
        1. Synthesize the strongest insights from all three into ONE superior answer.
           Do not just concatenate — genuinely integrate and elevate.
        2. Score the synthesis quality strictly from 1–10 (be honest, be harsh).
        3. List specific weaknesses that remain.
        4. Set needs_revision to true if score < 8, otherwise false.

        Respond with ONLY valid JSON — no markdown, no code fences:
        {{
            "synthesis": "<the full synthesized answer>",
            "score": <integer 1 to 10>,
            "needs_revision": <true or false>,
            "weaknesses": ["<weakness 1>", "<weakness 2>", ...]
        }}
    """)),
    # This message includes the original query + all 3 branch answers
    # {query} = user's original question
    # {analytical}, {creative}, {critical} = the 3 branch outputs from Stage 3
    ("human", textwrap.dedent("""\
        Original query: {query}

        ── ANALYTICAL PERSPECTIVE ──────────────────────────────────
        {analytical}

        ── CREATIVE PERSPECTIVE ────────────────────────────────────
        {creative}

        ── CRITICAL PERSPECTIVE ────────────────────────────────────
        {critical}

        Synthesize, score, and identify weaknesses:
    """)),
])

# critique_chain = Stage 4 full pipeline
# _critique_template formats the big prompt →
# model_precise generates the JSON (low temp for consistent JSON structure) →
# JsonOutputParser() parses the raw JSON string into a Python dict
# So critique_chain.invoke(...) returns a Python dict, not a string
critique_chain = _critique_template | model_precise | JsonOutputParser()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — CONSTITUTIONAL AI REVISION
#
# WHAT IT DOES:
#   Only runs when Stage 4 gives a score below 8.
#   Takes the synthesized draft answer and rewrites it according to 7 explicit
#   ethical/quality principles (Constitutional AI, inspired by Anthropic's technique).
#   Forces the model to fix every weakness identified in Stage 4.
#
# HOW IT WORKS:
#   CONSTITUTIONAL_PRINCIPLES is a Python list of principle strings.
#   In run_pipeline(), this list is formatted with bullet points and injected
#   into _revision_template via the {principles} variable.
#   The model then rewrites the draft, forced to follow all 7 principles.
#
# INPUT:  {"principles": ..., "query": ..., "synthesis": ..., "weaknesses": ...}
# OUTPUT: a plain string — the revised, improved answer
#
# RELATIONSHIP:
#   → Receives synthesis and weaknesses from Stage 4
#   → Output becomes final_answer if triggered
#   → If NOT triggered (score >= 8), final_answer = critique["synthesis"] directly
# ══════════════════════════════════════════════════════════════════════════════

# These 7 principles are applied to every answer when revision is needed.
# Each principle targets a specific quality failure that LLMs commonly make.
CONSTITUTIONAL_PRINCIPLES = [
    "Truthfulness    — Never assert something without sufficient evidence.",
    "Calibration     — Express uncertainty explicitly; hedge appropriately.",     # don't be overconfident
    "Helpfulness     — Give actionable, concrete, specific guidance the user can act on.",
    "Conciseness     — Cut every word that adds no meaning. Zero filler.",
    "Transparency    — Acknowledge limitations and what this answer cannot cover.",
    "Non-deception   — Avoid framing that could mislead, manipulate, or oversimplify.",
    "Completeness    — Address every identified weakness from the critique.",
]

# _revision_template sends 2 messages:
#   system = the constitutional principles + rules for the editor
#   human  = the original query + draft answer + list of weaknesses to fix
_revision_template = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""\
        You are a Constitutional AI editor — a master of precise, principled writing.
        Your task: revise the draft answer by strictly applying these principles:

        {principles}

        Non-negotiable rules:
        - You MUST address every identified weakness listed below.
        - Do NOT reduce depth or length — only improve quality and precision.
        - Output ONLY the revised answer text. No labels, no preamble.
    """)),
    # {synthesis}   = the Stage 4 draft answer to be improved
    # {weaknesses}  = the list of problems identified by Stage 4's self-critique
    ("human", textwrap.dedent("""\
        Query: {query}

        Draft answer:
        {synthesis}

        Weaknesses to resolve:
        {weaknesses}

        Write the constitutionally revised answer:
    """)),
])

# revision_chain = Stage 5 full pipeline
# _revision_template formats prompt →
# model_creative (higher temp=0.75) rewrites with fresh perspective →
# str_parser returns clean string
# Uses model_creative because revision benefits from some creative freedom
revision_chain = _revision_template | model_creative | str_parser


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — STRUCTURED OUTPUT EXTRACTION
#
# WHAT IT DOES:
#   Takes the final free-text answer and converts it into a clean, typed Python
#   object (AnalysisResult). The model fills in all the fields: summary,
#   key_insights, confidence_score, reasoning_quality, limitations, follow-up questions.
#
# HOW IT WORKS:
#   PydanticOutputParser reads the AnalysisResult schema and generates
#   "format instructions" — a text description of the JSON structure the model must follow.
#   These instructions are pre-filled into the template using .partial().
#   The model generates JSON → PydanticOutputParser validates it → returns AnalysisResult object.
#
#   Why .partial(format_instructions=...)?
#   The format_instructions string contains { and } characters (from the JSON schema).
#   If we directly embed it in the template string, LangChain would think those are
#   template variables and try to fill them in. .partial() pre-bakes the value BEFORE
#   template variable processing, so the braces are never interpreted as variables.
#
# INPUT:  {"query": ..., "final_answer": ...}
# OUTPUT: AnalysisResult object (the typed Pydantic model defined above)
#
# RELATIONSHIP:
#   → Receives final_answer from Stage 5 (if revision triggered) or Stage 4 (if not)
#   → Returns the AnalysisResult that run_pipeline() gives back to the caller
#   → display_result() reads from this object to print the formatted output
# ══════════════════════════════════════════════════════════════════════════════

# _pydantic_parser knows the structure of AnalysisResult and can:
#   1. Generate format instructions (text telling the model what JSON to produce)
#   2. Parse the model's JSON output into an AnalysisResult Python object
_pydantic_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

# .partial(format_instructions=...) pre-fills that one variable immediately,
# leaving {query} and {final_answer} to be filled at invoke() time.
# This avoids the JSON schema's braces being misread as LangChain variables.
_extraction_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise structured-data extractor.\n"
     "Convert the final answer into JSON matching this exact schema:\n\n"
     "{format_instructions}\n\n"          # filled by .partial() below — never changes
     "confidence_score: your honest probability that the answer is correct (0.0–1.0).\n"
     "reasoning_quality: your honest self-assessment of the chain of reasoning (1–10).\n"
     "Output ONLY valid JSON. No markdown, no code fences, no commentary."),
    ("human",
     "Query: {query}\n\n"                 # the user's original question
     "Final Answer:\n{final_answer}\n\n"  # the answer produced by stages 1-5
     "Extract the structured JSON now:"),
]).partial(format_instructions=_pydantic_parser.get_format_instructions())
# .get_format_instructions() generates a string like:
#   "The output should be a JSON object with these fields: summary (string), ..."

# extraction_chain = Stage 6 full pipeline
# _extraction_template formats prompt →
# model_precise generates the JSON (low temp for reliable JSON structure) →
# _pydantic_parser validates JSON and returns a real AnalysisResult Python object
extraction_chain = _extraction_template | model_precise | _pydantic_parser


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — _log()
#
# WHAT IT DOES:
#   Prints a formatted section header with a separator line, stage name,
#   and optional content. Used by run_pipeline() to print progress live.
#
# RELATIONSHIP:
#   → Called by run_pipeline() before and after each stage when verbose=True
#   → Not part of the AI pipeline — purely for terminal display
# ══════════════════════════════════════════════════════════════════════════════

def _log(stage: str, content: Any = "", width: int = 68):
    """
    Prints a formatted separator block to the terminal.

    Args:
      stage   : The stage label to display (e.g. "STAGE 1 › Meta-Prompt")
      content : Optional text to print below the header (model output preview)
      width   : Width of the separator line in characters
    """
    bar = "─" * width
    print(f"\n{bar}")
    print(f"  {stage}")       # indent the stage name for readability
    print(bar)
    if content:
        text = str(content)   # convert to string in case it's a dict or other type
        # Truncate very long outputs so the terminal doesn't flood
        if len(text) > 1400:
            text = text[:1400] + "\n  [... truncated for display ...]"
        print(text)


# ══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE — run_pipeline()
#
# WHAT IT DOES:
#   This is the MAIN FUNCTION that connects all 6 stages in the correct order.
#   You call this with a query string, and it returns an AnalysisResult.
#
# HOW IT WORKS (step by step):
#   1. Prints the query so you can confirm what is being processed
#   2. Calls meta_chain → gets system_prompt (Stage 1)
#   3. Calls cot_chain with system_prompt + query → gets cot_reasoning (Stage 2)
#   4. Calls tot_parallel with query + cot_reasoning → gets branches dict (Stage 3)
#   5. Calls critique_chain with query + 3 branches → gets critique dict (Stage 4)
#   6. Checks critique["needs_revision"]:
#        True  → calls revision_chain → gets revised final_answer (Stage 5)
#        False → uses critique["synthesis"] directly as final_answer
#   7. Calls extraction_chain with query + final_answer → returns AnalysisResult (Stage 6)
#
# Args:
#   query   : the user's question (string)
#   verbose : if True, prints each stage's output to terminal as it runs
#
# Returns:
#   AnalysisResult — a typed Python object with summary, insights, score, etc.
#
# RELATIONSHIP:
#   → Orchestrates: meta_chain, cot_chain, tot_parallel, critique_chain,
#                   revision_chain (conditional), extraction_chain
#   → Called by __main__ after capturing the user's query
#   → Its return value is passed to display_result()
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str, verbose: bool = True) -> AnalysisResult:
    width = 68
    # Print the query in a bold block so you can always see what is being processed
    # This is the first thing printed — confirms input was captured correctly
    print("\n" + "█" * width)
    print(f"  QUERY: {query}")
    print("█" * width)

    # ── Stage 1: Meta-Prompt ─────────────────────────────────────────────────
    # Pass query to meta_chain → LLM generates the best system prompt for this topic
    # system_prompt is a plain string, e.g.: "You are an expert in X, think step by step..."
    if verbose: _log("STAGE 1  ›  Meta-Prompt Generation  (LLM writes its own system prompt)")
    system_prompt = meta_chain.invoke({"query": query})
    if verbose: _log("Generated System Prompt", system_prompt)

    # ── Stage 2: Chain-of-Thought ────────────────────────────────────────────
    # Pass the Stage 1 system_prompt + original query to cot_chain
    # The model sees: [custom system prompt] + [few-shot examples] + [CoT instructions] + [query]
    # cot_reasoning is a long string with STEP 1 through STEP 5 labeled sections
    if verbose: _log("STAGE 2  ›  Chain-of-Thought Reasoning  (5-step structured decomposition)")
    cot_reasoning = cot_chain.invoke({"system_prompt": system_prompt, "query": query})
    if verbose: _log("CoT Output", cot_reasoning)

    # ── Stage 3: Tree of Thoughts ────────────────────────────────────────────
    # Send the same query + cot_reasoning to all 3 branches simultaneously
    # tot_parallel fires 3 API calls at the same time and collects results into a dict
    # branches = {"analytical": "...", "creative": "...", "critical": "..."}
    if verbose: _log("STAGE 3  ›  Tree of Thoughts  (Analytical + Creative + Critical in parallel)")
    branches = tot_parallel.invoke({"query": query, "cot_reasoning": cot_reasoning})
    if verbose:
        _log("Analytical Branch", branches["analytical"])
        _log("Creative Branch",   branches["creative"])
        _log("Critical Branch",   branches["critical"])

    # ── Stage 4: Self-Critique ───────────────────────────────────────────────
    # **branches unpacks the dict: passes analytical, creative, critical as separate keyword args
    # The model synthesizes all 3 branches and returns a JSON dict
    # critique["score"] tells us whether the synthesis is good enough
    if verbose: _log("STAGE 4  ›  Self-Critique & Synthesis  (model scores its own output)")
    critique = critique_chain.invoke({"query": query, **branches})
    score = critique.get("score", 0)   # default 0 if JSON parsing missed the field
    if verbose:
        _log(f"Synthesis  (score={score}/10  |  needs_revision={critique.get('needs_revision')})",
             critique.get("synthesis", ""))

    # ── Stage 5: Constitutional Revision (conditional) ───────────────────────
    # Only runs if the model rated its own synthesis below 8/10
    if critique.get("needs_revision", False):
        if verbose:
            _log(f"STAGE 5  ›  Constitutional AI Revision  (score {score} < 8 — triggered)")
        # Format CONSTITUTIONAL_PRINCIPLES list into a bullet-point string for the prompt
        final_answer = revision_chain.invoke({
            "principles": "\n".join(f"  • {p}" for p in CONSTITUTIONAL_PRINCIPLES),
            "query":      query,
            "synthesis":  critique["synthesis"],   # the draft to improve
            # Format weaknesses list into a dash-separated string
            "weaknesses": "\n".join(f"  – {w}" for w in critique.get("weaknesses", [])),
        })
        if verbose: _log("Constitutionally Revised Answer", final_answer)
    else:
        # Score was 8 or above — the synthesis is already good enough, skip revision
        if verbose:
            _log(f"STAGE 5  ›  Constitutional Revision skipped  (score {score} ≥ 8, no revision needed)")
        final_answer = critique["synthesis"]   # use Stage 4's answer directly

    # ── Stage 6: Structured Extraction ───────────────────────────────────────
    # Convert the final free-text answer into a typed AnalysisResult object
    # The model reads final_answer and fills in all the structured fields
    if verbose: _log("STAGE 6  ›  Structured Output Extraction  (Pydantic typed schema)")
    result = extraction_chain.invoke({"query": query, "final_answer": final_answer})
    # result is now an AnalysisResult Python object — fully typed and validated
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — STREAMING CHAT (stream_chat)
#
# WHAT IT DOES:
#   An interactive chatbot that streams tokens to the terminal in real time
#   (you see the AI typing word by word, not waiting for the full response).
#   Maintains full conversation history so the AI remembers previous messages.
#
# HOW IT WORKS:
#   - history list stores HumanMessage and AIMessage objects from previous turns
#   - Each loop iteration: appends user message → builds full messages list →
#     calls model.stream() → prints each token chunk as it arrives
#   - After streaming ends, saves the complete AI response to history
#   - Next turn: the model sees the entire conversation so far
#
# RELATIONSHIP:
#   → Completely independent from the 6-stage pipeline
#   → Uses model_creative (temp=0.75) for engaging, varied responses
#   → _STREAMING_SYSTEM is the advanced system prompt that shapes the AI's behavior
# ══════════════════════════════════════════════════════════════════════════════

# This system prompt shapes how the AI behaves in every chat response.
# It enforces: visible reasoning, quantified claims, uncertainty markers,
# structured headers for multi-part answers, and no hollow filler phrases.
_STREAMING_SYSTEM = textwrap.dedent("""\
    You are an expert-level AI assistant with mastery across science, technology,
    philosophy, business, and the humanities.

    Non-negotiable response rules:
    • Show a brief [REASONING] block before your answer — think out loud.
    • Use concrete examples, quantified claims ("~70% of cases"), and analogies.
    • Mark genuine uncertainty with [UNCERTAIN: <specific doubt>].
    • Structure answers with clear headers when answering multi-part questions.
    • End every response with: "Suggested next question: <one powerful follow-up>"
    • Never say "Great question!" or use hollow filler phrases.
""")


def stream_chat():
    """
    Interactive streaming chatbot with advanced system prompting and full history.

    How it works:
      1. history starts as an empty list
      2. User types a message → added to history as HumanMessage
      3. messages = [system_message] + history (full conversation context)
      4. model_creative.stream(messages) → returns a generator of token chunks
      5. Each chunk is printed immediately (no buffering) for real-time output
      6. Full response is saved to history as AIMessage
      7. Loop repeats — model sees all previous messages in next turn
      8. Type 'exit' to quit

    Relationship to pipeline:
      → This is the MODE 2 option (no 6 stages, just direct streaming chat)
      → Selected from __main__ when user enters 2
    """
    width = 68
    print("\n" + "═" * width)
    print("  ADVANCED STREAMING CHAT  (type 'exit' to quit)")
    print("═" * width)
    print("  Full message history maintained across the conversation.")
    print("═" * width)

    history: list = []  # stores the growing conversation as Message objects

    while True:
        user_input = input("\nYOU  ›  ").strip()

        # Exit keywords — any of these ends the chat
        if user_input.lower() in ("exit", "quit", "q"):
            print("\nGoodbye.")
            break

        # Ignore empty Enter presses — just loop again
        if not user_input:
            continue

        # Add the user's message to history so future turns remember it
        history.append(HumanMessage(content=user_input))

        # Build the full message list for this turn:
        # [system instructions] + [all past human + AI messages]
        messages = [SystemMessage(content=_STREAMING_SYSTEM)] + history

        # Start streaming — model_creative.stream() returns tokens one by one
        print("\nAI   ›  ", end="", flush=True)  # flush=True forces immediate print
        full_response = ""

        for chunk in model_creative.stream(messages):
            token = chunk.content         # the text fragment in this chunk
            print(token, end="", flush=True)  # print without newline, immediately
            full_response += token        # accumulate the full response

        # Save the complete AI reply to history for next turn's context
        history.append(AIMessage(content=full_response))
        print()  # newline after the streamed response ends


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY — display_result()
#
# WHAT IT DOES:
#   Takes the AnalysisResult object returned by run_pipeline() and prints
#   it in a clean, formatted way to the terminal.
#
# HOW IT WORKS:
#   Reads each field from the AnalysisResult Pydantic object and formats it:
#     - summary         → wrapped text paragraph
#     - key_insights    → numbered list
#     - confidence_score → visual progress bar with percentage
#     - reasoning_quality → visual progress bar with /10 score
#     - limitations       → bullet list
#     - follow_up_questions → arrow list
#
# RELATIONSHIP:
#   → Called after run_pipeline() finishes in __main__
#   → Reads from AnalysisResult (the output of Stage 6)
#   → No AI calls here — purely display logic
# ══════════════════════════════════════════════════════════════════════════════

def display_result(result: AnalysisResult):
    """
    Pretty-prints the final AnalysisResult to the terminal.

    Args:
      result : AnalysisResult object returned by run_pipeline()
    """
    width = 68
    print("\n" + "═" * width)
    print("  FINAL STRUCTURED RESULT")
    print("═" * width)

    # SUMMARY — wrap long text to fit the terminal width
    print(f"\nSUMMARY\n{'─' * width}")
    for line in textwrap.wrap(result.summary, width=width):
        print(f"  {line}")

    # KEY INSIGHTS — numbered list
    print(f"\nKEY INSIGHTS\n{'─' * width}")
    for i, insight in enumerate(result.key_insights, 1):
        print(f"  {i}. {insight}")

    # METRICS — ASCII progress bars for visual readability
    print(f"\nMETRICS\n{'─' * width}")
    # confidence_score is 0.0–1.0, multiply by 20 to get 0–20 filled blocks
    bar_conf = "█" * int(result.confidence_score * 20) + "░" * (20 - int(result.confidence_score * 20))
    # reasoning_quality is 1–10, use directly as filled block count
    bar_qual = "█" * result.reasoning_quality + "░" * (10 - result.reasoning_quality)
    print(f"  Confidence Score   [{bar_conf}]  {result.confidence_score:.0%}")
    print(f"  Reasoning Quality  [{bar_qual}]  {result.reasoning_quality}/10")

    # LIMITATIONS — bullet list of gaps in the answer
    print(f"\nLIMITATIONS\n{'─' * width}")
    for lim in result.limitations:
        print(f"  • {lim}")

    # FOLLOW-UP QUESTIONS — arrow list of suggested next questions
    print(f"\nSUGGESTED FOLLOW-UP QUESTIONS\n{'─' * width}")
    for q in result.follow_up_questions:
        print(f"  → {q}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — __main__ block
#
# WHAT IT DOES:
#   This block runs ONLY when the file is executed directly (python advanced_prompt_engine.py).
#   It is NOT executed if this file is imported as a module.
#
# HOW IT WORKS:
#   1. Prints the welcome banner
#   2. Loops until user enters a valid choice (1 or 2):
#        1 → Full 6-Stage Pipeline
#        2 → Streaming Chat
#   3. If choice 1:
#        - Loops until user types a non-empty query
#        - Calls run_pipeline(query) → AnalysisResult
#        - Calls display_result(result) to print it
#   4. If choice 2:
#        - Calls stream_chat() directly
#
# RELATIONSHIP:
#   → Calls run_pipeline() for option 1
#   → Calls stream_chat() for option 2
#   → Calls display_result() to show pipeline output
#   → This is where the user first enters their query
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    width = 68
    # Welcome banner — decorative only
    print("╔" + "═" * (width - 2) + "╗")
    print("║" + "  ADVANCED PROMPT ENGINEERING ENGINE".center(width - 2) + "║")
    print("║" + "  LangChain LCEL + Groq llama-3.3-70b-versatile".center(width - 2) + "║")
    print("╠" + "═" * (width - 2) + "╣")
    print("║" + "  1 — Full 6-Stage Pipeline".ljust(width - 2) + "║")
    print("║" + "      Meta-Prompt → CoT → Tree-of-Thoughts".ljust(width - 2) + "║")
    print("║" + "      Self-Critique → Constitutional AI → Pydantic".ljust(width - 2) + "║")
    print("║" + "  2 — Advanced Streaming Chat (with history)".ljust(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")

    # Loop until user types exactly "1" or "2" — prevents silent fallback to defaults
    while True:
        choice = input("\nEnter 1 or 2: ").strip()
        if choice in ("1", "2"):
            break
        print(f"  Invalid input '{choice}'. Please type 1 or 2.")

    if choice == "1":
        # Loop until user types an actual question — prevents empty query running default
        while True:
            query = input("\nEnter your query: ").strip()
            if query:
                break
            print("  Query cannot be empty. Please type a question.")

        # Run the full 6-stage pipeline and display the structured result
        result = run_pipeline(query, verbose=True)
        display_result(result)

    else:
        # choice == "2" — start the streaming chat
        stream_chat()
