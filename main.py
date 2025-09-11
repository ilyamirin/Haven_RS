import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

# LangGraph
from langgraph.graph import StateGraph, END

# Transformers (HF chat model)
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch


console = Console()


def read_agents(csv_path: str) -> List[Dict[str, Any]]:
    # Try multiple separators robustly
    for sep in [',', '\t', ';']:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            if len(df.columns) < 3:
                continue
            # Normalize columns
            cols = {c.lower().strip(): c for c in df.columns}
            # Map common names
            def pick(*names):
                for n in names:
                    if n in cols:
                        return cols[n]
                return None

            name_col = pick('агент', 'agent', 'name')
            role_col = pick('роль', 'role')
            persona_col = pick('личность', 'persona')
            tooling_col = pick('tooling', 'tools', 'инструменты', 'поддержка tooling?')
            prompt_col = pick('промпт', 'prompt', 'instruction', 'system')

            agents = []
            for _, row in df.iterrows():
                name = str(row[name_col]).strip() if name_col else str(row.iloc[0]).strip()
                role = str(row[role_col]).strip() if role_col else ''
                persona = str(row[persona_col]).strip() if persona_col else ''
                tooling = str(row[tooling_col]).strip() if tooling_col else ''
                prompt = str(row[prompt_col]).strip() if prompt_col else ''
                agents.append({
                    'name': name,
                    'role': role,
                    'persona': persona,
                    'tooling': tooling,
                    'prompt': prompt,
                })
            return agents
        except Exception:
            continue
    # Legacy fallback: no header, 3 columns
    df = pd.read_csv(csv_path, header=None)
    agents = []
    for _, row in df.iterrows():
        agents.append({
            'name': str(row[0]).strip(),
            'role': str(row[1]).strip() if len(row) > 1 else '',
            'persona': str(row[2]).strip() if len(row) > 2 else '',
            'tooling': '',
            'prompt': '',
        })
    return agents


def load_catalog(path: str) -> pd.DataFrame:
    # The file appears to be TSV (tab-separated)
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        # Fallback to comma
        return pd.read_csv(path)


def join_fields(row: pd.Series) -> str:
    # Join all fields to a single searchable text string
    parts = []
    for col, val in row.items():
        try:
            s = '' if pd.isna(val) else str(val)
        except Exception:
            s = ''
        parts.append(s)
    return ' '.join(parts)


def search_catalog(df: pd.DataFrame, query: str, top_k: int = 20) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    q = query.lower().strip()
    if not q:
        return df.head(top_k)
    # Precompute text corpus for scoring
    if '_joined_text' not in df.columns:
        df = df.copy()
        df['_joined_text'] = df.apply(join_fields, axis=1).str.lower()
    # Simple score: count of query keywords present
    keywords = [w for w in q.replace(',', ' ').split() if len(w) > 1]
    def score(text: str) -> int:
        return sum(1 for w in keywords if w in text)
    scores = df['_joined_text'].apply(score)
    df2 = df.assign(_score=scores)
    # Also prioritize by exact color/brand/name keyword matches if present
    for col in ['name', 'brand', 'category', 'color', 'colors', 'keyword', 'kinds']:
        if col in df2.columns:
            df2['_score'] += df2[col].astype(str).str.lower().apply(lambda t: 1 if any(w in t for w in keywords) else 0)
    df2 = df2.sort_values(by=['_score'], ascending=False)
    # Filter out zero-score unless everything is zero
    if df2['_score'].max() > 0:
        df2 = df2[df2['_score'] > 0]
    return df2.head(top_k)


class HFChat:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or os.environ.get('HF_MODEL', 'Qwen/Qwen3-4B')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
        )
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        if self.device == 'cpu':
            self.model = self.model.to('cpu')

    def stream_chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256):
        # Compose chat messages using tokenizer chat template if any
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        except Exception:
            # Fallback: simple concatenation
            prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")

        # Normalize inputs to a dict with input_ids and (optional) attention_mask
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            streamer=streamer,
        )

        import threading
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()


# GRAPH STATE
class RecoState(dict):
    # Keys:
    # - query: str
    # - catalog: pd.DataFrame
    # - shortlist: pd.DataFrame
    # - messages: List[Dict]
    # - final: Optional[Dict]
    pass


def make_agent_node(agent: Dict[str, Any], llm: HFChat):
    name = agent['name']
    persona = agent.get('persona', '')
    extra_prompt = agent.get('prompt', '')

    system_prompt = (
        f"Ты — {name}. {agent.get('role','')}\n"
        f"Личность: {persona}\n"
        f"Инструкция: {extra_prompt}\n"
        "Формат ответа в JSON со следующими полями: {\"thoughts\": string, \"answer\": string}."
    )

    def node(state: RecoState):
        query = state.get('query', '')
        # Provide a compact view of current shortlist (top 5 names/brands/colors)
        sl = state.get('shortlist')
        preview = []
        if isinstance(sl, pd.DataFrame) and not sl.empty:
            cols = [c for c in ['name', 'brand', 'category', 'color', 'colors'] if c in sl.columns]
            for _, r in sl.head(5).iterrows():
                preview.append({c: str(r.get(c, '')) for c in cols})
        user_prompt = (
            "Задача: помочь с рекомендацией товара по запросу клиента.\n"
            f"Запрос клиента: {query}\n"
            f"Текущий шорт-лист (до 5): {json.dumps(preview, ensure_ascii=False)}\n"
            "Дай полезные рассуждения (thoughts) и итоговый совет (answer)."
        )

        console.print(Panel(Text(f"[{name}] размышляет…", style="yellow"), title=f"{name}", border_style="yellow"))

        # Stream model output and try to parse JSON incrementally
        buffer = ''
        with Live(console=console, refresh_per_second=8):
            for tok in llm.stream_chat(system_prompt, user_prompt, max_new_tokens=300):
                buffer += tok
                # Live streaming: print tokens as they come (simple)
                console.print(tok, end="")
            console.print()  # newline

        # Try to extract JSON object from buffer
        thoughts = ''
        answer = buffer.strip()
        try:
            # find first '{' and last '}'
            start = buffer.find('{')
            end = buffer.rfind('}')
            if start != -1 and end != -1 and end > start:
                obj = json.loads(buffer[start:end+1])
                thoughts = str(obj.get('thoughts', '')).strip()
                answer = str(obj.get('answer', '')).strip() or answer
        except Exception:
            pass

        # Log nicely
        if thoughts:
            console.print(Panel(Text(thoughts, style="italic cyan"), title=f"{name} — мысли", border_style="cyan"))
        console.print(Panel(Text(answer, style="green"), title=f"{name} — ответ", border_style="green"))

        # Append to messages history
        msgs = state.get('messages', [])
        msgs.append({'agent': name, 'thoughts': thoughts, 'answer': answer})
        state['messages'] = msgs

        # Optionally: we could refine shortlist based on hints; to keep minimal, leave as-is
        return state

    return node


def make_coordinator_node(agent: Dict[str, Any], llm: HFChat):
    name = agent['name']
    persona = agent.get('persona', '')
    extra_prompt = agent.get('prompt', '')

    system_prompt = (
        f"Ты — {name}. Главный координатор. {agent.get('role','')}\n"
        f"Личность: {persona}\n"
        f"Инструкция: {extra_prompt}\n"
        "Твоя задача — выбрать лучший товар из шорт-листа."
        "Отвечай кратко: сначала рассуждения (thoughts), затем итог (answer)."
        "Итог в JSON: {\"selected_index\": int (0..N-1), \"rationale\": string}."
    )

    def node(state: RecoState):
        sl = state.get('shortlist')
        if not isinstance(sl, pd.DataFrame) or sl.empty:
            console.print("[red]Шорт-лист пуст — выбрать нечего.[/red]")
            return state
        # Build preview list for the coordinator
        cols = [c for c in ['name', 'brand', 'category', 'color', 'colors', 'mark', 'keyword'] if c in sl.columns]
        preview = []
        for _, r in sl.head(5).iterrows():
            preview.append({c: str(r.get(c, '')) for c in cols})

        user_prompt = (
            "Ниже кандидаты (до 5). Выбери один лучший и объясни кратко почему.\n"
            f"Кандидаты: {json.dumps(preview, ensure_ascii=False)}\n"
            "Верни JSON выбора."
        )

        console.print(Panel(Text(f"[{name}] анализирует кандидатов…", style="yellow"), title=f"{name}", border_style="yellow"))

        buffer = ''
        with Live(console=console, refresh_per_second=8):
            for tok in llm.stream_chat(system_prompt, user_prompt, max_new_tokens=256):
                buffer += tok
                console.print(tok, end="")
            console.print()

        # Parse JSON
        sel_idx = 0
        rationale = ''
        try:
            start = buffer.find('{')
            end = buffer.rfind('}')
            if start != -1 and end != -1 and end > start:
                obj = json.loads(buffer[start:end+1])
                sel_idx = int(obj.get('selected_index', 0))
                rationale = str(obj.get('rationale', ''))
        except Exception:
            pass

        # Bound index
        sel_idx = max(0, min(sel_idx, min(4, len(sl.index)-1)))
        selected_row = sl.head(5).iloc[sel_idx]
        console.print(Panel(Text(rationale or "Выбираю наиболее релевантный по запросу и описанию.", style="italic cyan"), title=f"{name} — мысли", border_style="cyan"))

        # Save final
        state['final'] = selected_row.to_dict()

        return state

    return node


def build_graph(agents: List[Dict[str, Any]], llm: HFChat):
    # Determine coordinator: Wardrobe Agent if present, else last
    coord_idx = None
    for i, a in enumerate(agents):
        if 'wardrobe agent' in a['name'].lower():
            coord_idx = i
            break
    if coord_idx is None:
        coord_idx = len(agents) - 1

    workflow = StateGraph(RecoState)

    # Prepare node: initial shortlist
    def prepare(state: RecoState):
        df = state.get('catalog')
        query = state.get('query', '')
        shortlist = search_catalog(df, query, top_k=20)
        state['shortlist'] = shortlist
        state['messages'] = []
        return state

    workflow.add_node('prepare', prepare)

    # Add helper agent nodes (all except coordinator)
    helper_names = []
    for i, ag in enumerate(agents):
        if i == coord_idx:
            continue
        node_name = f"agent_{i}"
        helper_names.append(node_name)
        workflow.add_node(node_name, make_agent_node(ag, llm))

    # Coordinator node
    workflow.add_node('coordinator', make_coordinator_node(agents[coord_idx], llm))

    # Edges: linear sequence
    workflow.set_entry_point('prepare')
    prev = 'prepare'
    for node_name in helper_names:
        workflow.add_edge(prev, node_name)
        prev = node_name
    workflow.add_edge(prev, 'coordinator')
    workflow.add_edge('coordinator', END)

    return workflow.compile()


def print_banner(model_name: str, query: str):
    console.print("\n" + "─" * 61)
    console.print("Мультиагентная рекомендация (LangGraph + Transformers)")
    console.print("Haven is ready to dress you. 👔🧥👢")
    console.print(f"Модель: {model_name}")
    console.print(f"Запрос пользователя: {query}")
    console.print("─" * 61)


def show_final(final_obj: Dict[str, Any]):
    if not final_obj:
        console.print("[red]Нет финальной рекомендации.[/red]")
        return
    fields = ['name', 'brand', 'category', 'color', 'colors', 'mark', 'keyword', 'kinds', 'description']
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Поле", style="dim", width=12)
    table.add_column("Значение")
    for f in fields:
        if f in final_obj and str(final_obj.get(f, '')).strip():
            table.add_row(f, str(final_obj.get(f, '')))
    console.print(Panel(table, title="Рекомендация координатора", border_style="magenta"))


def main():
    parser = argparse.ArgumentParser(description="Haven RS — мультиагентный рекомендатель (LangGraph + Transformers)")
    parser.add_argument('--query', type=str, default=None, help='Запрос пользователя (если не указан — будет задан интерактивно).')
    parser.add_argument('--model', type=str, default=None, help='HF модель (переопределяет переменную окружения HF_MODEL).')
    args = parser.parse_args()

    query = args.query
    if not query:
        console.print("Введите ваш запрос (например: Ищу теплую рубашку в клетку на осень):")
        try:
            query = input('> ').strip()
        except EOFError:
            query = ''
    if not query:
        console.print("[red]Пустой запрос. Выход.[/red]")
        sys.exit(1)

    agents = read_agents('agents_list.csv')
    if not agents:
        console.print("[red]Не удалось прочитать agents_list.csv[/red]")
        sys.exit(1)

    catalog = load_catalog(os.path.join('data', '27181_all_cards.csv'))
    if catalog is None or catalog.empty:
        console.print("[red]Не удалось прочитать каталог товаров data/27181_all_cards.csv[/red]")
        sys.exit(1)

    if args.model:
        os.environ['HF_MODEL'] = args.model
    llm = HFChat(os.environ.get('HF_MODEL', 'Qwen/Qwen3-4B'))

    print_banner(llm.model_name, query)

    graph = build_graph(agents, llm)
    state: RecoState = {
        'query': query,
        'catalog': catalog,
    }

    result_state = graph.invoke(state)
    if result_state is None:
        # Some LangGraph versions mutate the input state in place and return None
        result_state = state
    try:
        final_obj = result_state.get('final') if isinstance(result_state, dict) else None
    except Exception:
        final_obj = None
    show_final(final_obj)


if __name__ == '__main__':
    main()
