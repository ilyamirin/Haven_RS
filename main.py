#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Executable multi-agent product recommender using LangGraph + Hugging Face Transformers.
- Reads agents from agents_list.csv (last one should be Coordinator Agent).
- Reads catalog from data/27181_all_cards.csv (tab-separated).
- Accepts a natural language query and outputs one recommended product.
- Streams agents' colored thoughts to console using rich.

Prerequisites:
  1) Install Python deps (examples):
     pip install -r requirements.txt
  2) Ensure an HF model name is available (env HF_MODEL). Example defaults to Qwen/Qwen2.5-3B-Instruct.

Run:
  HF_MODEL="Qwen/Qwen2.5-3B-Instruct" python main.py --query "Ищу теплую рубашку в клетку на осень"
  or just: python main.py (you will be prompted)
"""

import argparse
import csv
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# LangGraph
from langgraph.graph import StateGraph, END

# Hugging Face Transformers for LLM
from transformers import AutoTokenizer, pipeline
import torch

# Rich for pretty streaming output
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table

MODEL_NAME = os.environ.get("HF_MODEL", "Qwen/Qwen-7B-Chat")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_CSV = os.path.join(ROOT_DIR, "agents_list.csv")
DATA_CSV = os.path.join(ROOT_DIR, "data", "27181_all_cards.csv")

console = Console()

# -----------------------------
# Utilities
# -----------------------------

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize_query(q: str) -> List[str]:
    q = normalize_text(q)
    # simple alnum tokens
    tokens = re.findall(r"[\wёа-я0-9]+", q, flags=re.IGNORECASE)
    return [t for t in tokens if t]


def load_agents(path: str) -> List[Dict[str, str]]:
    agents: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            agents.append({
                "name": row[0].strip(),
                "expertise": row[1].strip(),
                "collab": row[2].strip(),
            })
    return agents


def load_catalog(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        # The file is tab-separated based on the preview
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            items.append(row)
            if limit and len(items) >= limit:
                break
    return items


def score_item(item: Dict[str, Any], tokens: List[str]) -> float:
    # Combine fields that are likely useful
    fields = [
        item.get("name", ""), item.get("brand", ""), item.get("category", ""),
        item.get("keyword", ""), item.get("description", ""), item.get("colors", ""),
        item.get("text", ""),
    ]
    blob = normalize_text(" ".join(fields))
    score = 0.0
    for t in tokens:
        # simple term match + a little weight on name/brand
        if t in blob:
            score += 1.0
        if t in normalize_text(item.get("name", "")):
            score += 1.0
        if t in normalize_text(item.get("brand", "")):
            score += 0.5
        if t in normalize_text(item.get("category", "")):
            score += 0.5
    # small boost by rating if any
    try:
        mark = float(item.get("mark", "0") or 0)
        score += (mark - 3.0) * 0.2  # boost above neutral
    except Exception:
        pass
    return score


def shortlist_items(catalog: List[Dict[str, Any]], query: str, k: int = 20) -> List[Dict[str, Any]]:
    tokens = tokenize_query(query)
    scored = [(score_item(item, tokens), item) for item in catalog]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item for s, item in scored[:k] if s > 0 or k <= len(scored)]
    # Ensure not empty; if empty, take a few popular by mark
    if not top:
        catalog2 = sorted(catalog, key=lambda it: float(it.get("mark", "0") or 0), reverse=True)
        top = catalog2[:k]
    return top


def product_brief(item: Dict[str, Any]) -> str:
    # concise single-line summary to keep context small
    fields = [
        f"name={item.get('name','').strip()}",
        f"brand={item.get('brand','').strip()}",
        f"category={item.get('category','').strip()}",
        f"color={item.get('color','').strip() or item.get('colors','').strip()}",
        f"mark={item.get('mark','').strip()}",
    ]
    return ", ".join([f for f in fields if f])


# -----------------------------
# LLM Wrapper
# -----------------------------

@dataclass
class LLMFacade:
    model_name: str = MODEL_NAME
    temperature: float = 0.3

    # class-level cache to avoid reloading model multiple times
    _pipe = None
    _tokenizer = None

    def _ensure_loaded(self):
        if self.__class__._pipe is not None and self.__class__._tokenizer is not None:
            return
        # Load tokenizer and model with a local-first strategy; download from HF if missing
        # 1) Try local cache only
        try:
            tok = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True, trust_remote_code=True, local_files_only=True
            )
            local_only = True
        except Exception:
            local_only = False
            console.print(
                f"[yellow]Модель не найдена в локальном кэше. Загрузка с Hugging Face: [bold]{self.model_name}[/bold][/yellow]"
            )
            tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=True)

        # Prepare model
        try:
            from transformers import AutoModelForCausalLM  # lazy import to keep top-level clean
        except Exception as e:
            raise RuntimeError(f"Не удалось импортировать AutoModelForCausalLM: {e}")

        model_kwargs = {"trust_remote_code": True}
        dtype = None
        if torch.cuda.is_available():
            # Prefer GPU if available
            model_kwargs.update({"device_map": "auto"})
            dtype = torch.float16
        # Try to load model from local cache only first
        try:
            if local_only:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
                )
            else:
                # We already know it's not local; go ahead and download
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, trust_remote_code=True, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
                )
        except Exception:
            # If local-only attempt failed, fall back to download
            console.print(
                f"[yellow]Загружаем веса модели с Hugging Face (может занять время) — [bold]{self.model_name}[/bold][/yellow]"
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
            )

        # Build generation pipeline
        pipe_kwargs = {
            "model": model,
            "tokenizer": tok,
            "task": "text-generation",
            "trust_remote_code": True,
        }
        gen = pipeline(**pipe_kwargs, trust_remote_code=True)
        self.__class__._pipe = gen
        self.__class__._tokenizer = tok

    def is_available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()
        tok = self.__class__._tokenizer
        pipe = self.__class__._pipe
        # Use chat template if available
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback simple formatting
            prompt = (
                f"<|system|>\n{system_prompt}\n</s>\n<|user|>\n{user_prompt}\n</s>\n<|assistant|>\n"
            )
        max_new = 512
        outputs = pipe(
            prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=max(0.01, float(self.temperature)),
            top_p=0.95,
            eos_token_id=tok.eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
        )
        text = outputs[0]["generated_text"]
        # If pipeline returns the full prompt + continuation, strip the prompt prefix
        if text.startswith(prompt):
            text = text[len(prompt):]
        # Trim any stray end tokens
        text = text.strip()
        return text


# -----------------------------
# Multi-agent LangGraph
# -----------------------------

@dataclass
class RecoState:
    query: str
    catalog: List[Dict[str, Any]]
    shortlist: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Tuple[str, str]] = field(default_factory=list)  # (agent_name, text)
    suggestions: Dict[str, str] = field(default_factory=dict)  # agent_name -> suggestion text
    final_recommendation: Optional[Dict[str, Any]] = None


class AgentsOrchestrator:
    def __init__(self, agents: List[Dict[str, str]], llm: LLMFacade):
        if not agents:
            raise ValueError("No agents provided. Check agents_list.csv")
        self.agents = agents
        self.coordinator_name = agents[-1]["name"].strip()
        self.llm = llm
        self.colors = self._assign_colors([a["name"] for a in agents])

    def _assign_colors(self, names: List[str]) -> Dict[str, str]:
        palette = [
            "cyan", "magenta", "yellow", "green", "bright_blue",
            "bright_magenta", "bright_cyan", "bright_green", "bright_yellow",
            "orange1", "turquoise2",
        ]
        res = {}
        for i, n in enumerate(names):
            res[n] = palette[i % len(palette)]
        return res

    # Node: prepare shortlist
    def node_prepare(self, state: RecoState) -> RecoState:
        with console.status("[bold]Формирование шорт-листа товаров…"):
            state.shortlist = shortlist_items(state.catalog, state.query, k=12)
        return state

    # Nodes: individual agents
    def make_agent_node(self, agent: Dict[str, str]):
        name = agent["name"]
        expertise = agent["expertise"]
        collab = agent["collab"]
        color = self.colors.get(name, "white")

        system_prompt = f"""
Вы — {name}. Ваша экспертиза: {expertise}. Коллаборация: {collab}.
Вы работаете в мультиагентной системе рекомендаций. Контекст общий.
Задача: изучить запрос пользователя и краткий список кандидатов и предложить, кого рекомендовать и почему.
Формат вывода:
- Критерии и соображения (коротко)
- 1-2 подходящих кандидата с кратким обоснованием в одну строку на каждого (укажите name или brand)
- Возможные риски или проверку для координатора
Пишите по-русски, сжато.
""".strip()

        def node(state: RecoState) -> RecoState:
            # Build a compact shortlist text
            shortlist_str = "\n".join(
                [f"{i+1}. {product_brief(it)}" for i, it in enumerate(state.shortlist[:6])]
            )
            user_prompt = f"""
Пользовательский запрос: {state.query}
Кандидаты (сокращённый список):
{shortlist_str}
Сделайте свои рекомендации.
""".strip()
            try:
                reply = self.llm.generate(system_prompt, user_prompt)
            except Exception as e:
                reply = f"[Ошибка LLM: {e}]"
            state.suggestions[name] = reply
            state.messages.append((name, reply))
            # stream to console immediately
            console.print(Panel.fit(reply, title=name, border_style=color))
            return state

        return node

    # Node: Coordinator
    def node_coordinator(self, state: RecoState) -> RecoState:
        name = self.coordinator_name
        color = self.colors.get(name, "white")

        # Build combined suggestions
        combined = []
        for agent_name, text in state.suggestions.items():
            if agent_name == name:
                continue
            combined.append(f"[{agent_name}]\n{text}")
        combined_text = "\n\n".join(combined)
        shortlist_text = "\n".join([product_brief(it) for it in state.shortlist[:8]])

        system_prompt = f"""
Вы — {name}. Вы координируете агентов и принимаете финальное решение.
Цель: выбрать один лучший товар для рекомендации пользователю с учётом запроса и мнений агентов.
Критерии: соответствие запросу, качество (mark), уместность бренда/категории/цвета.
Выберите ровно один товар из SHORTLIST и верните JSON с ключами: index, name, reason.
Где index — индекс позиции в SHORTLIST (начиная с 1) вашего выбора, reason — краткое обоснование.
Не добавляйте ничего кроме JSON.
""".strip()
        user_prompt = f"""
USER QUERY:
{state.query}

SHORTLIST (нумерация с 1):
{shortlist_text}

AGENTS SUGGESTIONS:
{combined_text}
""".strip()
        try:
            reply = self.llm.generate(system_prompt, user_prompt)
        except Exception as e:
            reply = f"{{\n  \"index\": 1, \"name\": \"fallback\", \"reason\": \"Ошибка LLM: {e}\"\n}}"

        # show coordinator thoughts
        console.print(Panel.fit(reply, title=name, border_style=color))

        # Parse JSON robustly
        chosen_index = 1
        try:
            import json
            data = json.loads(reply)
            idx = int(data.get("index", 1))
            if 1 <= idx <= len(state.shortlist):
                chosen_index = idx
        except Exception:
            # try to find index like "index": 3
            m = re.search(r"index\"?\s*[:=]\s*(\d+)", reply)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= len(state.shortlist):
                    chosen_index = idx
        # set final
        state.final_recommendation = state.shortlist[chosen_index - 1]
        return state

    def build_graph(self) -> StateGraph:
        graph = StateGraph(RecoState)
        graph.add_node("prepare", self.node_prepare)
        # add agent nodes for all except coordinator
        for agent in self.agents:
            if agent["name"].strip() == self.coordinator_name:
                continue
            graph.add_node(agent["name"], self.make_agent_node(agent))
        graph.add_node("coordinator", self.node_coordinator)

        # Edges: prepare -> each agent (sequentially), then -> coordinator
        graph.set_entry_point("prepare")
        prev = "prepare"
        for agent in self.agents:
            if agent["name"].strip() == self.coordinator_name:
                continue
            graph.add_edge(prev, agent["name"])
            prev = agent["name"]
        graph.add_edge(prev, "coordinator")
        graph.add_edge("coordinator", END)
        return graph


# -----------------------------
# CLI and main
# -----------------------------

def ensure_files_exist():
    missing = []
    if not os.path.exists(AGENTS_CSV):
        missing.append(AGENTS_CSV)
    if not os.path.exists(DATA_CSV):
        missing.append(DATA_CSV)
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))


def run(query: str):
    ensure_files_exist()
    agents = load_agents(AGENTS_CSV)
    if not agents:
        console.print("[red]agents_list.csv пуст или невалиден[/red]")
        sys.exit(1)

    catalog = load_catalog(DATA_CSV)
    if not catalog:
        console.print("[red]Каталог пуст или не удалось прочитать data/27181_all_cards.csv[/red]")
        sys.exit(1)

    # Intro
    console.rule("Мультиагентная рекомендация (LangGraph + Transformers)")
    console.print(f"Модель: [bold]{MODEL_NAME}[/bold]")
    console.print(f"Запрос пользователя: [bold]{query}[/bold]\n")

    llm = LLMFacade(model_name=MODEL_NAME, temperature=0.3)
    if not llm.is_available():
        console.print("[yellow]Внимание:[/yellow] LLM (Transformers) недоступен/не загружен.\n"
                      "Убедитесь, что установлены зависимости и указан корректный HF_MODEL:\n"
                      "  pip install -r requirements.txt\n"
                      "  set HF_MODEL=Qwen/Qwen2.5-3B-Instruct\n"
                      "  # либо другой поддерживаемый чатовый LLM на Hugging Face\n")

    orchestrator = AgentsOrchestrator(agents, llm)
    state = RecoState(query=query, catalog=catalog)
    graph = orchestrator.build_graph().compile()

    # Execute the graph
    result = graph.invoke(state)

    # Final output
    if isinstance(result, dict):
        recommended = result.get("final_recommendation") or {}
    else:
        recommended = getattr(result, "final_recommendation", None) or {}

    if not recommended:
        console.print("[red]Не удалось сформировать рекомендацию[/red]")
        sys.exit(2)

    console.rule("Рекомендация координатора")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Поле")
    table.add_column("Значение")
    def add_row(k: str, v: Any):
        if v is None:
            v = ""
        v = str(v)
        if len(v) > 250:
            v = v[:247] + "…"
        table.add_row(k, v)

    add_row("name", recommended.get("name"))
    add_row("brand", recommended.get("brand"))
    add_row("category", recommended.get("category"))
    add_row("color", recommended.get("color") or recommended.get("colors"))
    add_row("mark", recommended.get("mark"))
    add_row("keyword", recommended.get("keyword"))
    add_row("kinds", recommended.get("kinds"))
    add_row("description", recommended.get("description"))
    console.print(table)

    console.print("\n[green]Готово.[/green]")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent product recommender (LangGraph + Transformers)")
    parser.add_argument("--query", "-q", type=str, help="Пользовательский запрос на естественном языке")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    q = args.query
    if not q:
        try:
            q = input("Введите ваш запрос: ").strip()
        except KeyboardInterrupt:
            print()
            sys.exit(130)
    if not q:
        console.print("[red]Пустой запрос[/red]")
        sys.exit(1)
    run(q)
