"""Part C: for each paragraph of the Education lecture, generate one catch-up question grounded in that paragraph.

The script has no timestamps, so a moment in the video is mapped to a paragraph by its share of the
19-minute running time (words are assumed to be spoken at a constant rate). This is an approximation.
"""
import json
import os
import re
import zipfile
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "data" / "raw" / "wandering_eyes" / "Videos & Tests" / "education_script.docx"
OUT = ROOT / "results" / "catchup_questions.json"
VIDEO_S = 19 * 60

PROMPT = """You are a tutor in a video lecture. The student's attention drifted while the lecturer said the passage below.
Write ONE short question (max 25 words) that brings the student back to the key idea of exactly this passage.
It should invite the student to think, not test trivia. Then give a two-sentence answer key that uses only the passage.
Reply as JSON: {"question": "...", "answer": "..."}

Passage:
"""


def paragraphs() -> list[str]:
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf-8")
    paras = [re.sub(r"<[^>]+>", "", p).strip() for p in xml.split("</w:p>")]
    paras = [p for p in paras if p][2:]  # drop title and word-count line
    # merge unnumbered continuation paragraphs into the numbered one before them
    merged = []
    for p in paras:
        if re.match(r"^\d+\.", p) or not merged:
            merged.append(p)
        else:
            merged[-1] += " " + p
    return merged


def timeline(paras: list[str]) -> list[dict]:
    words = [len(p.split()) for p in paras]
    total, acc, out = sum(words), 0, []
    for i, (p, w) in enumerate(zip(paras, words)):
        start = acc / total * VIDEO_S
        acc += w
        out.append({"index": i + 1, "start_s": round(start, 1), "end_s": round(acc / total * VIDEO_S, 1),
                    "text": re.sub(r"^\d+\.\s*", "", p)})
    return out


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    items = timeline(paragraphs())
    for it in items:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT + it["text"]}])
        it |= json.loads(r.choices[0].message.content)
        print(it["index"], it["question"])
    OUT.write_text(json.dumps({"model": "gpt-4o-mini", "temperature": 0, "items": items}, indent=1, ensure_ascii=False))
