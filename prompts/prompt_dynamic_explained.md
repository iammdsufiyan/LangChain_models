# prompt_dynamic.py — Line by Line Explanation

## Full Code

```python
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    temperature=0.7,
    max_new_tokens=150
)
model = ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')

paper_input = st.selectbox("Select Research Paper Name", [...])
style_input = st.selectbox("Select Explanation Style", [...])
length_input = st.selectbox("Select Explanation Length", [...])

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
```

---

## Line-by-Line Explanation

### Imports

```python
from langchain_core.prompts import PromptTemplate, load_prompt
```
- `PromptTemplate` — LangChain class to create prompt templates with `{placeholder}` variables.
- `load_prompt` — loads a saved prompt template from a JSON file (like `template.json`) instead of writing it in code.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
```
- `HuggingFaceEndpoint` — connects to a HuggingFace-hosted model via their Inference API (needs `HUGGINGFACEHUB_API_TOKEN` in `.env`).
- `ChatHuggingFace` — wraps the endpoint so it behaves like a **chat model** (accepts messages, returns `.content`).

```python
from dotenv import load_dotenv
```
- Loads environment variables from a `.env` file (e.g., your HuggingFace API key).

```python
import streamlit as st
```
- Streamlit is the UI framework — it turns Python scripts into interactive web apps.

---

### Loading Environment Variables

```python
load_dotenv()
```
- Reads the `.env` file and injects variables like `HUGGINGFACEHUB_API_TOKEN` into the environment so HuggingFace client can authenticate.

---

### Setting Up the LLM

```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    temperature=0.7,
    max_new_tokens=150
)
```
- `repo_id` — the HuggingFace model to use. Here it's **Qwen 2.5 7B Instruct**, a capable instruction-tuned model.
- `task="conversational"` — tells HuggingFace this is a chat/instruction task.
- `temperature=0.7` — controls randomness. `0` = deterministic, `1` = very creative. `0.7` is a balanced setting.
- `max_new_tokens=150` — the model will generate at most 150 tokens in its response.

```python
model = ChatHuggingFace(llm=llm)
```
- Wraps the `HuggingFaceEndpoint` into a chat-compatible interface.
- Now `model` can accept formatted prompts and return a response with a `.content` attribute.

---

### Streamlit UI

```python
st.header('Reasearch Tool')
```
- Renders a large heading on the web page: **"Reasearch Tool"** (note: typo in original code).

```python
paper_input = st.selectbox("Select Research Paper Name", [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
])
```
- Creates a dropdown menu for the user to pick a research paper.
- The selected value is stored in `paper_input`.

```python
style_input = st.selectbox("Select Explanation Style", [
    "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"
])
```
- Another dropdown to choose HOW the summary should be written.
- Stored in `style_input`.

```python
length_input = st.selectbox("Select Explanation Length", [
    "Short (1-2 paragraphs)",
    "Medium (3-5 paragraphs)",
    "Long (detailed explanation)"
])
```
- Dropdown to choose HOW LONG the summary should be.
- Stored in `length_input`.

---

### Loading the Prompt Template

```python
template = load_prompt('template.json')
```
- Reads `template.json` from disk and reconstructs a `PromptTemplate` object.
- The template contains `{paper_input}`, `{style_input}`, `{length_input}` placeholders.
- This is the "dynamic" part — the prompt is **externalized** to a file, not hardcoded.

---

### Running the Chain on Button Click

```python
if st.button('Summarize'):
```
- Renders a **"Summarize"** button. The code inside only runs when the user clicks it.

```python
    chain = template | model
```
- The `|` (pipe) operator is LangChain's way to chain steps together.
- It means: first format the prompt using `template`, then send it to `model`.
- This creates a `RunnableSequence`: `PromptTemplate → ChatHuggingFace`.

```python
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
```
- `.invoke()` runs the full chain in one call.
- It first fills in the placeholders in the template, then sends the final prompt to the model.
- The model's response is stored in `result` (an `AIMessage` object).

```python
    st.write(result.content)
```
- `.content` extracts the text from the `AIMessage` object.
- `st.write()` displays it on the Streamlit page.

---

## Writing the Same Code WITHOUT the Chain (`|` operator)

Instead of using `chain = template | model` and `chain.invoke(...)`, you manually do each step:

**Step 1** — Format the prompt yourself using `.format_messages()`  
**Step 2** — Pass the formatted messages directly to the model using `.invoke()`

```python
from langchain_core.prompts import load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    temperature=0.7,
    max_new_tokens=150
)
model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox("Select Research Paper Name", [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
])

style_input = st.selectbox("Select Explanation Style", [
    "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"
])

length_input = st.selectbox("Select Explanation Length", [
    "Short (1-2 paragraphs)",
    "Medium (3-5 paragraphs)",
    "Long (detailed explanation)"
])

template = load_prompt('template.json')

if st.button('Summarize'):

    # Step 1: Format the prompt — fill in the placeholders manually
    formatted_messages = template.format_messages(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )

    # Step 2: Send the formatted prompt directly to the model
    result = model.invoke(formatted_messages)

    # Step 3: Display the response
    st.write(result.content)
```

---

## Chain vs No-Chain — Side by Side

| | With Chain (`\|`) | Without Chain |
|---|---|---|
| **Style** | Declarative (pipeline) | Imperative (step by step) |
| **Code** | `chain = template \| model` then `chain.invoke({...})` | `formatted = template.format_messages({...})` then `model.invoke(formatted)` |
| **Output** | Same `AIMessage` object | Same `AIMessage` object |
| **Flexibility** | Easy to add more steps (`\| parser \| validator`) | More control, easier to debug each step |
| **Best for** | Production pipelines | Learning / debugging |

Both approaches produce **exactly the same result**. The chain is just a cleaner shorthand.
