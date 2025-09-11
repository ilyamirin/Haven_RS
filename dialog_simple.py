import os
import sys
import argparse
import json
from typing import Dict, Any, List, Optional

import pandas as pd

# We will reuse catalog/agents utilities from main.py to keep changes minimal
from main import read_agents, load_catalog, search_catalog

# LangChain (for prompt construction) + Transformers (for streaming)
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch


class HFStreamer:
    """
    Minimal streaming chat wrapper over HuggingFace transformers.
    We still use LangChain's ChatPromptTemplate to construct prompts.
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or os.environ.get('HF_MODEL', 'Qwen/Qwen3-4B-Instruct-2507')
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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        except Exception:
            prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")

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


def find_wardrobe_agent(agents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for a in agents:
        if 'wardrobe agent' in str(a.get('name', '')).lower():
            return a
    return agents[0] if agents else None


def print_stream(tokens_iter):
    buffer = ''
    for tok in tokens_iter:
        buffer += tok
        sys.stdout.write(tok)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return buffer


def build_system_prompt(agent: Dict[str, Any], instruction: str) -> str:
    name = agent.get('name', 'Wardrobe Agent')
    persona = agent.get('persona', '')
    role = agent.get('role', '')
    extra = agent.get('prompt', '')
    return (
        f"Ты — {name}. {role}\n"
        f"Личность: {persona}\n"
        f"Инструкция: {extra}\n"
        f"{instruction}"
    )


def clarification_loop(agent: Dict[str, Any], llm: HFStreamer, query: str, catalog: pd.DataFrame, max_rounds: int = 3) -> str:
    refined_query = query.strip()
    for _ in range(max_rounds):
        shortlist = search_catalog(catalog, refined_query, top_k=5)
        preview = []
        if isinstance(shortlist, pd.DataFrame) and not shortlist.empty:
            cols = [c for c in ['name', 'brand', 'category', 'color', 'colors'] if c in shortlist.columns]
            for _, r in shortlist.head(5).iterrows():
                preview.append({c: str(r.get(c, '')) for c in cols})

        system_prompt = build_system_prompt(agent, (
            "Твоя задача сейчас — задать до 3 уточняющих вопросов пользователю, чтобы уточнить запрос к каталогу.\n"
            "На каждом шаге оцени, достаточно ли информации.\n"
            "Формат ответа в JSON: {\"thoughts\": string, \"question\": string, \"need_more\": bool, \"refined_query\": string}.\n"
            "Если информации достаточно, установи need_more=false и сформулируй refined_query."
        ))
        human = (
            "Твой контекст:\n"
            f"Текущий запрос: {refined_query}\n"
            f"Превью кандидатов (до 5): {json.dumps(preview, ensure_ascii=False)}\n"
            "Сначала сгенерируй размышления (thoughts), затем задай 1 короткий вопрос (question).\n"
            "Верни строго один JSON-объект."
        )
        # Use LangChain to build the prompt text (for formal compliance)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            ("human", "{human}"),
        ]).format(system=system_prompt, human=human)

        print(f"\n[{agent.get('name','Wardrobe Agent')}] задаёт уточняющий вопрос…")
        buffer = print_stream(llm.stream_chat(system_prompt, human, max_new_tokens=280))

        thoughts = ''
        question = ''
        need_more = True
        candidate_query = refined_query
        try:
            start = buffer.find('{')
            end = buffer.rfind('}')
            if start != -1 and end != -1 and end > start:
                obj = json.loads(buffer[start:end+1])
                thoughts = str(obj.get('thoughts', '')).strip()
                question = str(obj.get('question', '')).strip()
                need_more = bool(obj.get('need_more', True))
                rq = str(obj.get('refined_query', '')).strip()
                if rq:
                    candidate_query = rq
        except Exception:
            question = buffer.strip()

        if thoughts:
            print(f"{agent.get('name','Wardrobe Agent')} — мысли:\n{thoughts}")
        if question:
            print(f"{agent.get('name','Wardrobe Agent')} — вопрос:\n{question}")

        print("Ваш ответ (Enter — пропустить/завершить):")
        try:
            user_reply = input('> ').strip()
        except EOFError:
            user_reply = ''

        if user_reply:
            refined_query = (candidate_query + ' ' + user_reply).strip()
        else:
            refined_query = candidate_query

        if not need_more or not user_reply:
            break

    return refined_query or query


def final_recommendation(agent: Dict[str, Any], llm: HFStreamer, refined_query: str, catalog: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
    shortlist = search_catalog(catalog, refined_query, top_k=top_k)
    items_preview = []
    if isinstance(shortlist, pd.DataFrame) and not shortlist.empty:
        # Collect compact dict for LLM context
        for _, r in shortlist.iterrows():
            d = {}
            for col, val in r.items():
                try:
                    s = '' if pd.isna(val) else str(val)
                except Exception:
                    s = ''
                if s:
                    d[col] = s
            items_preview.append(d)

    system_prompt = build_system_prompt(agent, (
        "Проанализируй список кандидатов и предложи лучшую рекомендацию пользователю.\n"
        "Верни JSON: {\"thoughts\": string, \"answer\": string}."
    ))
    human = (
        f"Итоговый уточнённый запрос: {refined_query}\n"
        f"Кандидаты (до {top_k}): {json.dumps(items_preview[:top_k], ensure_ascii=False)}\n"
        "Сначала подумай (thoughts), затем дай краткий совет (answer). Верни один JSON-объект."
    )

    print(f"\n[{agent.get('name','Wardrobe Agent')}] формирует рекомендацию…")
    buffer = print_stream(llm.stream_chat(system_prompt, human, max_new_tokens=350))

    thoughts = ''
    answer = buffer.strip()
    try:
        start = buffer.find('{')
        end = buffer.rfind('}')
        if start != -1 and end != -1 and end > start:
            obj = json.loads(buffer[start:end+1])
            thoughts = str(obj.get('thoughts', '')).strip()
            answer = str(obj.get('answer', '')).strip() or answer
    except Exception:
        pass

    if thoughts:
        print(f"{agent.get('name','Wardrobe Agent')} — мысли:\n{thoughts}")
    if answer:
        print(f"{agent.get('name','Wardrobe Agent')} — ответ:\n{answer}")

    # Return also raw shortlist to allow plain printing of items
    return {
        'thoughts': thoughts,
        'answer': answer,
        'shortlist': shortlist if isinstance(shortlist, pd.DataFrame) else pd.DataFrame(),
    }


def print_results_table(df: pd.DataFrame, top_k: int = 10):
    if df is None or df.empty:
        print("\nНичего не найдено по вашему запросу.")
        return
    print("\nТоп результатов по каталогу:")
    cols_order = [c for c in ['name', 'brand', 'category', 'color', 'colors', 'mark', 'keyword', 'kinds', 'description'] if c in df.columns]
    extra_cols = [c for c in df.columns if c not in cols_order and not c.startswith('_')]
    cols = cols_order + extra_cols
    for idx, (_, row) in enumerate(df.head(top_k).iterrows(), start=1):
        print(f"\n#{idx}")
        for c in cols:
            try:
                val = row.get(c, '')
            except Exception:
                val = ''
            if pd.isna(val) or str(val).strip() == '':
                continue
            print(f"{c}: {val}")


def print_banner(model_name: str, query: str):
    print("\n" + "-" * 61)
    print("Простой диалог с Wardrobe Agent (LangChain + Transformers)")
    print("Haven is ready to dress you. 👔🧥👢")
    print(f"Модель: {model_name}")
    print(f"Начальный запрос: {query}")
    print("-" * 61)


def main():
    parser = argparse.ArgumentParser(description="Простой диалог: Wardrobe Agent уточняет запрос и рекомендует товары")
    parser.add_argument('--query', type=str, default=None, help='Начальный запрос пользователя')
    parser.add_argument('--model', type=str, default=None, help='HF модель (по умолчанию Qwen/Qwen3-4B-Instruct-2507)')
    parser.add_argument('--max_clarify', type=int, default=3, help='Максимум уточняющих раундов Wardrobe Agent')
    parser.add_argument('--top_k', type=int, default=10, help='Сколько результатов каталога показать')
    args = parser.parse_args()

    query = args.query
    if not query:
        print("Введите ваш запрос (например: Ищу теплую рубашку в клетку на осень):")
        try:
            query = input('> ').strip()
        except EOFError:
            query = ''
    if not query:
        print("Пустой запрос. Выход.")
        sys.exit(1)

    # Ensure default model per requirements
    model_name = args.model or os.environ.get('HF_MODEL') or 'Qwen/Qwen3-4B-Instruct-2507'
    os.environ['HF_MODEL'] = model_name

    agents = read_agents('agents_list.csv')
    if not agents:
        print("Не удалось прочитать agents_list.csv")
        sys.exit(1)

    wardrobe = find_wardrobe_agent(agents)
    if not wardrobe:
        print("Wardrobe Agent не найден в agents_list.csv")
        sys.exit(1)

    catalog = load_catalog(os.path.join('data', '27181_all_cards.csv'))
    if catalog is None or catalog.empty:
        print("Не удалось прочитать каталог товаров data/27181_all_cards.csv")
        sys.exit(1)

    llm = HFStreamer(model_name)

    print_banner(model_name, query)

    refined_query = clarification_loop(wardrobe, llm, query, catalog, max_rounds=args.max_clarify)
    outcome = final_recommendation(wardrobe, llm, refined_query, catalog, top_k=args.top_k)

    # Show catalog results
    shortlist = outcome.get('shortlist')
    if isinstance(shortlist, pd.DataFrame):
        print_results_table(shortlist, top_k=args.top_k)


if __name__ == '__main__':
    main()
