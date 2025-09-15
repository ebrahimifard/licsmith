import json
from pathlib import Path
from typing import Dict, Optional, List

import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


class QAResult(BaseModel):
    """Strict shape the model must return."""
    answers: Dict[str, str]
    summary: Optional[str] = None
    confidence: Optional[float] = None


class LicenseAnalyzerLLM:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        context_reserve_tokens: int = 1024,
        prefer_4bit: bool = True
    ):
        self.QUESTIONS: List[str] = [
            "What type of license is this (MIT, GPL, Apache, BSD, etc.)?",
            "Can this software be used for commercial purposes?",
            "Does this license require attribution when using the software?",
            "Can the source code be modified?",
            "Can modified versions be redistributed?",
            "Does this license have copyleft requirements (must share source code of derivatives)?",
            "What are the main restrictions or obligations?",
            "Is this license compatible with proprietary/closed-source software?",
            "Does this license provide patent protection?",
            "What happens if someone violates this license?"
        ]

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_reserve_tokens = max(512, context_reserve_tokens)
        self.prefer_4bit = prefer_4bit and (self.device == "cuda")

        self.MAX_NEW_TOKENS = 700
        self.TEMPERATURE = 0.0
        self.TOP_P = 1.0
        self.RANDOM_SEED = 42

        print(f"🔄 Loading model: {self.model_name}")
        print(f"📱 Device: {self.device}")

        self._load_model_and_tokenizer()
        print("✅ Model loaded successfully!")

    def _load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(trust_remote_code=True)
        if self.prefer_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["device_map"] = "auto"
                print("🧮 Using 4-bit quantization")
            except Exception as e:
                print(f"⚠️  4-bit quantization unavailable: {e}. Falling back to torch_dtype auto.")
                load_kwargs["torch_dtype"] = "auto"
                load_kwargs["device_map"] = "auto" if self.device == "cuda" else None
        else:
            load_kwargs["torch_dtype"] = "auto"
            load_kwargs["device_map"] = "auto" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model_max_len = getattr(self.model.config, "max_position_embeddings", 4096) or 4096

    def _build_prompt(self, license_text: str) -> str:
        instruction = (
            "You are a careful legal assistant. Read the LICENSE TEXT and answer the fixed questionnaire.\n"
            "Return ONLY valid JSON, no prose, no markdown fences.\n"
            "Be concise and quote specific obligations when relevant.\n"
            "If the license does not say, write a best-effort short answer (you may say it's unclear)."
        )

        skeleton = {
            "answers": {q: "<fill in>" for q in self.QUESTIONS},
            "summary": "<optional 2-4 sentence summary>",
            "confidence": 0.0
        }
        schema = json.dumps(skeleton, indent=2)

        user_content = (
            "LICENSE TEXT:\n"
            f"{license_text}\n\n"
            "QUESTIONNAIRE (answer every item in the JSON under 'answers' using the question text as the key):\n"
            + "\n".join([f"- {q}" for q in self.QUESTIONS]) +
            "\n\nReturn JSON with this exact shape and keys:\n" + schema
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            return (
                f"<|system|>\n{instruction}\n<|end|>\n"
                f"<|user|>\n{user_content}\n<|end|>\n"
                f"<|assistant|>\n"
            )

    def _prepare_inputs(self, license_text: str) -> str:
        max_input_tokens = max(
            512, self.model_max_len - self.context_reserve_tokens - self.MAX_NEW_TOKENS
        )
        tokens = self.tokenizer(
            license_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_tokens,
            return_attention_mask=False,
            return_tensors=None,
        )
        if len(tokens["input_ids"]) < max_input_tokens:
            return license_text
        return self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

    def analyze(self, license_file_path: str) -> QAResult:
        set_seed(self.RANDOM_SEED)

        p = Path(license_file_path)
        if not p.exists():
            raise FileNotFoundError(f"License file not found: {p}")
        text = p.read_text(encoding="utf-8", errors="ignore")

        text = self._prepare_inputs(text)
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        print("Hi1")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
                # temperature=self.TEMPERATURE,
                # top_p=self.TOP_P,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        generated = full_text[len(prompt_text):].strip()

        if generated.startswith("```"):
            generated = generated.strip().lstrip("`")
            if generated.lower().startswith("json"):
                generated = generated[4:].strip()
            if "```" in generated:
                generated = generated.split("```", 1)[0].strip()

        start = generated.find("{")
        end = generated.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Model did not return JSON. Raw:\n{generated[:800]}")

        raw_json = generated[start:end+1]
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            repaired = raw_json.replace(",\n}", "\n}").replace(",\n]", "\n]")
            data = json.loads(repaired)

        if "answers" not in data:
            data["answers"] = {}
        for q in self.QUESTIONS:
            data["answers"].setdefault(q, "Unclear from the provided license text.")

        return QAResult(**data)

    def print_results(self, result: QAResult):
        print("\n" + "="*80)
        print("🔍 LICENSE ANALYSIS RESULTS (LLM-only)")
        print("="*80)
        for i, q in enumerate(self.QUESTIONS, 1):
            ans = result.answers.get(q, "")
            print(f"\n{i}. {q}\n   📝 {ans}\n" + "-"*60)
        if result.summary:
            print("\n📌 Summary:\n", result.summary)
        if result.confidence is not None:
            try:
                print(f"\nConfidence (self-reported): {float(result.confidence):.2f}")
            except Exception:
                print(f"\nConfidence (self-reported): {result.confidence}")

    def save_results(self, result: QAResult, license_file: Path):
        out_path = license_file.with_suffix(".llm-license-review.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        print(f"💾 JSON saved to: {out_path}")


# -------------- Run directly --------------

if __name__ == "__main__":
    analyzer = LicenseAnalyzerLLM()
    license_path = input("Enter path to LICENSE file: ").strip()
    result = analyzer.analyze(license_path)
    analyzer.print_results(result)
    analyzer.save_results(result, Path(license_path))
