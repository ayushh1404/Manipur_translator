import os
import json
from pathlib import Path
from openai import OpenAI


_pricing_cache = None


def _default_pricing_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "openai_pricing.json"


def _load_pricing_config() -> dict:
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache
    pricing_path = os.getenv("OPENAI_PRICING_FILE") or str(
        _default_pricing_path()
    )
    path_obj = Path(pricing_path)
    if not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parents[1] / path_obj
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            _pricing_cache = json.load(f)
    except Exception:
        _pricing_cache = {}
    return _pricing_cache


def _get_model_pricing(model_name: str | None) -> dict | None:
    if not model_name:
        return None
    config = _load_pricing_config()
    tier = os.getenv("OPENAI_PRICE_TIER") or config.get(
        "default_tier") or "standard"
    tiers = config.get("tiers", {})
    tier_pricing = tiers.get(tier, {})
    if not tier_pricing:
        return None
    aliases = config.get("aliases", {})
    if model_name in aliases:
        model_name = aliases[model_name]
    for key in sorted(tier_pricing.keys(), key=len, reverse=True):
        if model_name == key or model_name.startswith(key):
            return tier_pricing[key]
    return None


def calculate_token_cost(usage: dict | None, model_name: str | None) -> dict | None:
    if not usage:
        return None
    pricing = _get_model_pricing(model_name)
    if not pricing:
        return None
    input_price = pricing.get("input_per_million")
    output_price = pricing.get("output_per_million")
    cached_input_price = pricing.get("cached_input_per_million")
    if input_price is None or output_price is None:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    if isinstance(cached_tokens, int) and cached_input_price is not None:
        non_cached_tokens = max(prompt_tokens - cached_tokens, 0)
        cached_input_cost = (cached_tokens / 1_000_000) * cached_input_price
        non_cached_input_cost = (non_cached_tokens / 1_000_000) * input_price
        input_cost = cached_input_cost + non_cached_input_cost
    else:
        cached_tokens = None
        cached_input_cost = None
        non_cached_input_cost = None
        input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost
    return {
        "model": model_name,
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": cached_tokens,
        },
        "pricing_per_million": {
            "input": input_price,
            "cached_input": cached_input_price,
            "output": output_price,
        },
        "input_cost": input_cost,
        "cached_input_cost": cached_input_cost,
        "non_cached_input_cost": non_cached_input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

# ═══════════════════════════════════════════════════════════════════
# LAZY INITIALIZATION
# ═══════════════════════════════════════════════════════════════════


_openai_client = None


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def answer_from_kb(
    query: str,
    chunks: list[str],
    include_raw: bool = False,
    user_prompt: str | None = None,
    user_persona: dict | None = None,
    conversation_history: list | None = None,
    kb_topics: list[str] | None = None,
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
):
    """
    Generate answer from knowledge base chunks using GPT

    Args:
        query: User's question
        chunks: List of relevant text chunks from KB

    Returns:
        Generated answer
    """
    openai = get_openai_client()

    context = "\n\n---\n\n".join(chunks)

    user_prompt_clean = (user_prompt or "").strip()
    user_persona_clean = user_persona if isinstance(
        user_persona, dict) else None
    history_clean = conversation_history if isinstance(
        conversation_history, list) else []
    topics_clean = [t for t in (kb_topics or [])
                    if isinstance(t, str) and t.strip()]
    # if user_prompt_clean:
    #     system_prompt = (
    #         f"{user_prompt_clean}\n"
    #         "\n"
    #         "CRITICAL INSTRUCTIONS - FOLLOW STRICTLY:\n"
    #         "1. ROLE ADHERENCE:\n"
    #         "   - You MUST strictly follow the role, persona, tone, and behavior defined above\n"
    #         "   - Maintain the expertise level and communication style specified in your role\n"
    #         "   - Stay in character at all times\n"
    #         "\n"
    #         "2. FACTUAL ACCURACY:\n"
    #         "   - Answer ONLY using the provided context for factual claims\n"
    #         "   - NEVER invent or hallucinate information\n"
    #         "   - If factual info is not in context, acknowledge this professionally\n"
    #         "\n"
    #         "3. CONVERSATION FLOW:\n"
    #         "   - ALWAYS analyze the COMPLETE conversation history\n"
    #         "   - Understand context, references, and what was discussed before\n"
    #         "   - Maintain conversational continuity across messages\n"
    #         "\n"
    #         "4. FOLLOW-UP QUESTIONS:\n"
    #         "   When user says: 'yes', 'no', 'tell me more', 'elaborate', 'what about X', 'can you explain that':\n"
    #         "   - Treat these as natural conversation continuations\n"
    #         "   - Reference the previous discussion naturally\n"
    #         "   - Continue the topic seamlessly\n"
    #         "   - NEVER say 'not in knowledge base' for follow-ups to previous answers\n"
    #         "\n"
    #         "5. SIMPLE ACKNOWLEDGMENTS:\n"
    #         "   When user says: 'hi', 'hello', 'yes', 'no', 'ok', 'thanks', 'thank you':\n"
    #         "   - Respond naturally and conversationally in your defined role\n"
    #         "   - If it's a follow-up to previous discussion, continue that topic\n"
    #         "   - NEVER treat these as KB lookup failures\n"
    #         "   - Be warm, professional, and helpful\n"
    #         "\n"
    #         "6. ELABORATION REQUESTS:\n"
    #         "   When user asks to elaborate on something you mentioned:\n"
    #         "   - Expand on that specific topic using the context\n"
    #         "   - Provide more details, examples, or explanation\n"
    #         "   - Reference what you said before naturally\n"
    #         "\n"
    #         "7. INFORMATION UNAVAILABILITY:\n"
    #         "   - ONLY say info is unavailable for genuinely NEW factual topics not in context\n"
    #         "   - Never say this for conversational responses or follow-ups\n"
    #         "   - When info is truly unavailable, be helpful and suggest alternatives\n"
    #         "\n"
    #         "8. TRANSPARENCY:\n"
    #         "   - Never mention 'knowledge base', 'context', 'system instructions', or 'chunks'\n"
    #         "   - Never break character or expose your internal workings\n"
    #         "   - Communicate naturally as if having a real conversation\n"
    #     )
    #     if user_persona_clean:
    #         system_prompt += f"\n\nUser persona context:\n{user_persona_clean}\n"
    # else:
    #     system_prompt = (
    #         "You are a knowledgeable, helpful, and professional assistant.\n"
    #         "\n"
    #         "CRITICAL INSTRUCTIONS - FOLLOW STRICTLY:\n"
    #         "1. FACTUAL ACCURACY:\n"
    #         "   - Answer ONLY using the provided context for factual claims\n"
    #         "   - NEVER invent or hallucinate information\n"
    #         "   - If factual info is not in context, acknowledge this professionally\n"
    #         "\n"
    #         "2. CONVERSATION FLOW:\n"
    #         "   - ALWAYS analyze the COMPLETE conversation history\n"
    #         "   - Understand context, references, and what was discussed before\n"
    #         "   - Maintain conversational continuity across messages\n"
    #         "\n"
    #         "3. FOLLOW-UP QUESTIONS:\n"
    #         "   When user says: 'yes', 'no', 'tell me more', 'elaborate', 'what about X', 'can you explain that':\n"
    #         "   - Treat these as natural conversation continuations\n"
    #         "   - Reference the previous discussion naturally\n"
    #         "   - Continue the topic seamlessly\n"
    #         "   - NEVER say 'not in knowledge base' for follow-ups to previous answers\n"
    #         "\n"
    #         "4. SIMPLE ACKNOWLEDGMENTS:\n"
    #         "   When user says: 'hi', 'hello', 'yes', 'no', 'ok', 'thanks', 'thank you':\n"
    #         "   - Respond naturally and conversationally\n"
    #         "   - If it's a follow-up to previous discussion, continue that topic\n"
    #         "   - NEVER treat these as KB lookup failures\n"
    #         "   - Be warm, professional, and helpful\n"
    #         "\n"
    #         "5. ELABORATION REQUESTS:\n"
    #         "   When user asks to elaborate on something you mentioned:\n"
    #         "   - Expand on that specific topic using the context\n"
    #         "   - Provide more details, examples, or explanation\n"
    #         "   - Reference what you said before naturally\n"
    #         "\n"
    #         "6. INFORMATION UNAVAILABILITY:\n"
    #         "   - ONLY say info is unavailable for genuinely NEW factual topics not in context\n"
    #         "   - Never say this for conversational responses or follow-ups\n"
    #         "   - When info is truly unavailable:\n"
    #         "     * Be helpful and professional\n"
    #         "     * If KB topics available, suggest related topics you can help with\n"
    #         "     * Otherwise, offer to help with questions from available sources\n"
    #     )
    #     if topics_clean:
    #         system_prompt += f"\n\nAvailable topics:\n{', '.join(topics_clean[:5])}\n"
    #     if user_persona_clean:
    #         system_prompt += f"\n\nUser persona context:\n{user_persona_clean}\n"
    #     system_prompt += (
    #         "\n"
    #         "7. TRANSPARENCY:\n"
    #         "   - Never mention 'knowledge base', 'context', 'system instructions', or 'chunks'\n"
    #         "   - Never break character or expose your internal workings\n"
    #         "   - Communicate naturally as if having a real conversation\n"
    #         "\n"
    #         "8. STYLE:\n"
    #         "   - Be concise, confident, and professional\n"
    #         "   - Use natural, conversational language\n"
    #         "   - Engage meaningfully with the user\n"
    #     )

    # history_block = ""
    # if history_clean:
    #     lines = []
    #     for item in history_clean[-8:]:
    #         if not isinstance(item, dict):
    #             continue
    #         role = item.get("role", "")
    #         content = item.get("content", "")
    #         if role == "ai":
    #             role = "assistant"
    #         if not role or not content:
    #             continue
    #         lines.append(f"{role}: {str(content)[:300]}")
    #     if lines:
    #         history_block = "\n".join(lines)

    # resp = openai.chat.completions.create(
    #     model=(model or "gpt-4o-mini"),
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": (
    #                 f"Conversation history:\n{history_block if history_block else '[none]'}\n\n"
    #                 f"Context:\n{context}\n\n"
    #                 f"Question:\n{query}"
    #             ),
    #         },
    #     ],
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )

    # answer = resp.choices[0].message.content.strip()
    # if include_raw:
    #     return answer, resp.model_dump()
    # return answer
    if user_prompt_clean:
        # When user_prompt exists, it IS the system prompt
        # We only add MINIMAL, NON-CONFLICTING technical instructions
        system_prompt = f"{user_prompt_clean}\n\n"

        # Only add these MINIMAL technical constraints that don't conflict with user instructions:
        system_prompt += (
            "═══════════════════════════════════════════════════════════\n"
            "TECHNICAL CONSTRAINTS (Do not let these override your instructions above):\n"
            "═══════════════════════════════════════════════════════════\n"
            "1. For factual claims, use only the provided context below\n"
            "2. Read the full conversation history to understand what was already discussed\n"
            "3. Never mention 'knowledge base', 'context', 'chunks', or technical internals to users\n"
            "4. GREETING RULES:\n"
            "   - ONLY use greetings (Hey, Hi, Hello) for the FIRST message in a conversation\n"
            "   - If conversation history exists, DO NOT start with greetings\n"
            "   - Mid-conversation: Start directly with your answer/question\n"
            "   - Example: 'Skalix can help by...' NOT 'Hey! Skalix can help by...'\n"
            "   - If you do greet, match the persona/tone defined above\n"
        )

        if user_persona_clean:
            system_prompt += f"\n\nAdditional context about the user:\n{user_persona_clean}\n"

    else:
        # Default professional assistant behavior when no user_prompt provided
        system_prompt = (
            "You are a knowledgeable, helpful, and professional assistant.\n"
            "\n"
            "INSTRUCTIONS:\n"
            "1. Answer using ONLY the provided context for factual claims - never invent information\n"
            "2. Always analyze the COMPLETE conversation history to understand context\n"
            "3. For follow-up questions (like 'yes', 'no', 'tell me more', 'elaborate'):\n"
            "   - Reference the previous conversation naturally\n"
            "   - Continue the discussion seamlessly\n"
            "   - Never say 'not in knowledge base' for conversational continuations\n"
            "4. For greetings or simple acknowledgments ('hi', 'hello', 'yes', 'no', 'ok', 'thanks'):\n"
            "   - Respond naturally and conversationally\n"
            "   - If it's a follow-up to previous discussion, continue that topic\n"
            "   - Never treat these as KB lookup failures\n"
            "5. When user asks to elaborate on something you mentioned earlier:\n"
            "   - Expand on that specific topic using the context\n"
            "   - Provide more details, examples, or explanation\n"
            "6. ONLY say information is unavailable for genuinely NEW factual topics not in context\n"
            "   - Never say this for conversational responses or follow-ups\n"
            "7. Never mention 'knowledge base', 'context', or 'system instructions' to users\n"
            "8. GREETING BEHAVIOR:\n"
            "   - ONLY use greetings (Hey, Hi, Hello) for the FIRST message in a conversation\n"
            "   - If conversation history exists, DO NOT start responses with greetings\n"
            "   - Mid-conversation: Start directly with your answer or question\n"
            "   - Example: 'Revenue Intelligence helps by...' NOT 'Hey! Revenue Intelligence helps...'\n"
            "   - Keep responses professional and to the point\n"
            "9. Be concise, confident, and professional\n"
        )

        # Add KB topics information if available
        if topics_clean:
            system_prompt += f"\n\nAvailable knowledge base topics:\n{', '.join(topics_clean[:5])}\n"

        if user_persona_clean:
            system_prompt += f"\n\nUser persona context:\n{user_persona_clean}\n"

    # ═══════════════════════════════════════════════════════════════════
    # BUILD CONVERSATION HISTORY - Keep FULL content for proper context
    # ═══════════════════════════════════════════════════════════════════

    history_block = ""
    if history_clean:
        lines = []
        # Use last 12 messages for comprehensive context
        for item in history_clean[-12:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            content = item.get("content", "")
            if role == "ai":
                role = "assistant"
            if not role or not content:
                continue
            # Keep FULL content - critical for understanding follow-ups and user info collection
            lines.append(f"{role}: {str(content)}")
        if lines:
            history_block = "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    # BUILD USER MESSAGE - Make it crystal clear what needs to be analyzed
    # ═══════════════════════════════════════════════════════════════════

    user_message_parts = []

    # 1. Conversation history (most important for understanding context)
    if history_block:
        user_message_parts.append(
            f"CONVERSATION HISTORY (Read carefully to understand what's been discussed and what user info you have):\n{history_block}"
        )
    else:
        user_message_parts.append(
            "CONVERSATION HISTORY:\n[This is the start of the conversation]")

    # 2. Context from KB
    if context:
        user_message_parts.append(
            f"\nKNOWLEDGE BASE CONTEXT (Use this for answering factual questions):\n{context}"
        )
    else:
        user_message_parts.append(
            "\nKNOWLEDGE BASE CONTEXT:\n[No relevant context found - if user asks about information not in above history, you don't have it]")

    # 3. Current user query (what they're saying right now)
    user_message_parts.append(
        f"\nCURRENT USER MESSAGE:\n{query}\n\n"
        "---\n"
        "Now respond according to your instructions. Remember:\n"
        "- Follow your conversation flow exactly as instructed\n"
        "- Check the conversation history to see what user info you've already collected\n"
        "- Continue your information gathering process if instructed to do so\n"
        "- For follow-up questions about previous topics, elaborate naturally\n"
    )

    user_message = "\n".join(user_message_parts)

    # ═══════════════════════════════════════════════════════════════════
    # CALL OPENAI API
    # ═══════════════════════════════════════════════════════════════════

    resp = openai.chat.completions.create(
        model=(model or "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    answer = resp.choices[0].message.content.strip()

    if include_raw:
        return answer, resp.model_dump()
    return answer
