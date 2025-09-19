from pydantic import BaseModel
from typing import Dict, Optional
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

class QAResult(BaseModel):
    """Strict shape the model must return."""
    answers: Dict[str, str]
    summary: Optional[str] = None
    confidence: Optional[float] = None


class LLMWrapper:
    def __init__(self, license_file_path):
        self.license_file_path = license_file_path

    def launch_model(self):
        start_time = time.time()
        self._load_model_and_tokenizer()
        self.model_load_time = time.time() - start_time

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
                if self.verbose:
                    print("🧮 Using 4-bit quantization")
            except Exception as e:
                if self.verbose:
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
            try:
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_content},
                ]
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except Exception:
                # Fallback if chat template fails
                pass
        
        # Fallback prompt format
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

    def analyse(self, license_file_path: str) -> QAResult:
        """Analyse a license file and return structured results."""
        set_seed(self.RANDOM_SEED)

        p = Path(license_file_path)
        if not p.exists():
            raise FileNotFoundError(f"License file not found: {p}")
        
        print(f"📄 Reading license file: {license_file_path}")
        text = p.read_text(encoding="utf-8", errors="ignore")
        print(f"📊 License file loaded ({len(text)} characters)")

        text = self._prepare_inputs(text)
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        print("🤖 Generating analysis...")
        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                repetition_penalty=1.05,
            )
        inference_time = time.time() - start_time
        print(f"⏱️  Analysis completed in {inference_time:.2f}s")

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        generated = full_text[len(prompt_text):].strip()

        # Clean up markdown fences if present
        if generated.startswith("```"):
            generated = generated.strip().lstrip("`")
            if generated.lower().startswith("json"):
                generated = generated[4:].strip()
            if "```" in generated:
                generated = generated.split("```", 1)[0].strip()

        # Extract JSON from the response
        start = generated.find("{")
        end = generated.rfind("}")
        if start == -1 or end == -1 or end <= start:
            lines = generated.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if '{' in line and not in_json:
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if '}' in line and in_json:
                    break
            
            if json_lines:
                generated = '\n'.join(json_lines)
                start = generated.find("{")
                end = generated.rfind("}")
            
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"Model did not return JSON. Raw output:\n{generated[:800]}")

        raw_json = generated[start:end+1]
        
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            print("⚠️  JSON parsing error, attempting repair...")
            repaired = raw_json.replace(",\n}", "\n}").replace(",\n]", "\n]").replace('",\n  }', '"\n  }')
            try:
                data = json.loads(repaired)
                print("✅ JSON repaired successfully")
            except json.JSONDecodeError:
                print("❌ Could not parse JSON, creating fallback response")
                data = {
                    "answers": {q: "Unable to parse model response" for q in self.QUESTIONS},
                    "summary": "Error occurred during analysis",
                    "confidence": 0.0
                }

        if "answers" not in data:
            data["answers"] = {}
        
        for q in self.QUESTIONS:
            if q not in data["answers"]:
                data["answers"][q] = "Unclear from the provided license text."

        return QAResult(**data)

