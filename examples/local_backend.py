from __future__ import annotations

import json
import math
import re
import sys
from typing import Any


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def extract_original(prompt: str) -> tuple[str, str | None]:
    original_match = re.search(r"Original task:\s*(.*?)\s*Proposed answer:\s*(.*)\s*$", prompt, re.DOTALL)
    if not original_match:
        return prompt, None
    return original_match.group(1).strip(), original_match.group(2).strip()


def solve_arithmetic(text: str) -> str | None:
    normalized = text.replace("×", "*").replace("÷", "/")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", normalized)
    if not match:
        return None
    left = float(match.group(1))
    right = float(match.group(3))
    operator = match.group(2)
    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    else:
        if right == 0.0:
            return "undefined"
        result = left / right
    if result.is_integer():
        return str(int(result))
    return format(result, ".12g")


def solve_sentiment(text: str) -> str | None:
    lowered = text.casefold()
    positive = {"love", "excellent", "wonderful", "great", "happy", "fantastic", "good"}
    negative = {"hate", "terrible", "awful", "bad", "sad", "horrible", "poor"}
    positive_score = sum(word in lowered for word in positive)
    negative_score = sum(word in lowered for word in negative)
    if positive_score == negative_score == 0:
        return None
    return "positive" if positive_score >= negative_score else "negative"


def solve_email(text: str) -> str | None:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def solve_city_country(text: str) -> str | None:
    match = re.search(r"for\s+([A-Za-zÀ-ž .'-]+),\s*([A-Za-zÀ-ž .'-]+)", text, re.IGNORECASE)
    if not match:
        return None
    city = match.group(1).strip()
    country = re.split(r"[.\n]", match.group(2).strip())[0].strip()
    return json.dumps({"city": city, "country": country}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def solve_boolean(text: str) -> str | None:
    lowered = text.casefold()
    match = re.search(r"is\s+(-?\d+)\s+(?:greater than|>)\s+(-?\d+)", lowered)
    if match:
        return "true" if int(match.group(1)) > int(match.group(2)) else "false"
    match = re.search(r"is\s+(-?\d+)\s+(?:less than|<)\s+(-?\d+)", lowered)
    if match:
        return "true" if int(match.group(1)) < int(match.group(2)) else "false"
    return None


def solve(text: str, task_type: str) -> str:
    task_text, _ = extract_original(text)
    solvers = {
        "question_answering": (solve_arithmetic, solve_boolean),
        "classification": (solve_sentiment, solve_boolean),
        "extraction": (solve_email,),
        "structured": (solve_city_country,),
    }
    for solver in solvers.get(task_type, ()):
        result = solver(task_text)
        if result is not None:
            return result
    arithmetic = solve_arithmetic(task_text)
    if arithmetic is not None:
        return arithmetic
    email = solve_email(task_text)
    if email is not None:
        return email
    sentiment = solve_sentiment(task_text)
    if sentiment is not None:
        return sentiment
    city_country = solve_city_country(task_text)
    if city_country is not None:
        return city_country
    boolean = solve_boolean(task_text)
    if boolean is not None:
        return boolean
    return task_text.strip().splitlines()[-1].strip()


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def confidence_response(prompt: str, task_type: str, truth: bool) -> str:
    original, proposed = extract_original(prompt)
    expected = solve(original, task_type)
    correct = proposed is not None and normalized(proposed) == normalized(expected)
    probability = 0.995 if correct else 0.005
    if truth:
        return json.dumps(
            {
                "correct_probability": probability,
                "incorrect_probability": 1.0 - probability,
            },
            separators=(",", ":"),
        )
    return json.dumps({"confidence": probability}, separators=(",", ":"))


def tokens(text: str, probability: float) -> list[dict[str, Any]]:
    values = []
    cursor = 0
    for position, match in enumerate(re.finditer(r"\S+\s*", text)):
        token = match.group(0)
        start = cursor
        cursor += len(token)
        values.append(
            {
                "token": token,
                "logprob": math.log(probability),
                "probability": probability,
                "position": position,
                "start_char": start,
                "end_char": cursor,
            }
        )
    return values


def main() -> None:
    request = json.load(sys.stdin)
    prompt = request["prompt"]["user"]
    metadata = request.get("metadata") or {}
    task_type = str(metadata.get("task_type", "question_answering"))
    purpose = str(metadata.get("purpose", "baseline"))
    if purpose == "forced_confidence":
        text = confidence_response(prompt, task_type, False)
    elif purpose == "truth_verification":
        text = confidence_response(prompt, task_type, True)
    else:
        text = solve(prompt, task_type)
    response = {
        "model": request.get("model", "reference-engine-v1"),
        "text": text,
        "finish_reason": "stop",
        "token_probabilities": tokens(text, 0.995),
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(prompt.split()) + len(text.split()),
            "estimated_cost": 0.0,
            "currency": "USD",
        },
        "reproducibility": {
            "engine": "reference-engine-v1",
            "seed": request.get("seed"),
            "task_type": task_type,
            "purpose": purpose,
        },
    }
    json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
