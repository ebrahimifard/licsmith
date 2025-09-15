import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import pandas as pd
from tabulate import tabulate


class QAResult(BaseModel):
    """Strict shape the model must return."""
    answers: Dict[str, str]
    summary: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ModelPerformance:
    """Track model performance metrics."""
    model_name: str
    works: bool
    load_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: str = ""
    result: Optional[QAResult] = None


class LicenseAnalyzerLLM:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        context_reserve_tokens: int = 1024,
        prefer_4bit: bool = True,
        verbose: bool = True
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
        self.verbose = verbose

        self.MAX_NEW_TOKENS = 700
        self.TEMPERATURE = 0.0
        self.TOP_P = 1.0
        self.RANDOM_SEED = 42

        if self.verbose:
            print(f"🔄 Loading model: {self.model_name}")
            print(f"📱 Device: {self.device}")

        start_time = time.time()
        self._load_model_and_tokenizer()
        self.load_time = time.time() - start_time
        
        if self.verbose:
            print(f"✅ Model loaded successfully in {self.load_time:.2f}s!")

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

    def analyze(self, license_file_path: str) -> Tuple[QAResult, float]:
        set_seed(self.RANDOM_SEED)

        p = Path(license_file_path)
        if not p.exists():
            raise FileNotFoundError(f"License file not found: {p}")
        text = p.read_text(encoding="utf-8", errors="ignore")

        text = self._prepare_inputs(text)
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
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
            repaired = raw_json.replace(",\n}", "\n}").replace(",\n]", "\n]").replace('",\n  }', '"\n  }')
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
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

        return QAResult(**data), inference_time

    def get_memory_usage(self) -> float:
        """Get approximate memory usage in MB."""
        if hasattr(self, 'model'):
            try:
                if self.device == "cuda":
                    return torch.cuda.memory_allocated() / 1024 / 1024
                else:
                    # Rough estimate for CPU
                    return sum(p.nelement() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
            except:
                return 0.0
        return 0.0


class ModelComparison:
    """Compare multiple models on license analysis."""
    
    # Comprehensive list of models to test
    MODELS_TO_TEST = [
        "microsoft/Phi-3-mini-4k-instruct",     # 3.8B - Microsoft's latest small model
        "microsoft/phi-2",                       # 2.7B - Proven performer
        "microsoft/DialoGPT-medium",            # 117M - Fast and lightweight
        "Qwen/Qwen2.5-3B-Instruct",            # 3B - Alibaba's strong model
        "Qwen/Qwen2.5-7B-Instruct",            # 7B - Larger Qwen model
        "HuggingFaceH4/zephyr-7b-beta",        # 7B - Community favorite
        "mistralai/Mistral-7B-Instruct-v0.2",  # 7B - Mistral's instruction model
        "google/flan-t5-base",                  # 250M - Google's T5 variant
        "microsoft/DialoGPT-small",             # 117M - Even smaller option
    ]
    
    def __init__(self, use_4bit: bool = True):
        self.use_4bit = use_4bit
        self.results: List[ModelPerformance] = []
    
    def test_model(self, model_name: str, license_file_path: str) -> ModelPerformance:
        """Test a single model."""
        print(f"\n🧪 Testing {model_name}...")
        
        performance = ModelPerformance(model_name=model_name, works=False)
        
        try:
            # Test basic loading and generation
            start_time = time.time()
            analyzer = LicenseAnalyzerLLM(
                model_name=model_name,
                prefer_4bit=self.use_4bit,
                verbose=False
            )
            performance.load_time = time.time() - start_time
            performance.memory_usage_mb = analyzer.get_memory_usage()
            
            # Test actual analysis
            result, inference_time = analyzer.analyze(license_file_path)
            performance.inference_time = inference_time
            performance.result = result
            performance.works = True
            
            print(f"✅ {model_name} - Load: {performance.load_time:.1f}s, Inference: {performance.inference_time:.1f}s")
            
        except Exception as e:
            performance.error_message = str(e)[:100]  # Truncate long errors
            print(f"❌ {model_name} - Error: {performance.error_message}")
        
        return performance
    
    def run_comparison(self, license_file_path: str) -> List[ModelPerformance]:
        """Run comparison across all models."""
        print("🚀 Starting model comparison...")
        print(f"📄 License file: {license_file_path}")
        print(f"🔧 Using 4-bit quantization: {self.use_4bit}")
        print(f"📱 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        self.results = []
        
        for model_name in self.MODELS_TO_TEST:
            try:
                performance = self.test_model(model_name, license_file_path)
                self.results.append(performance)
                
                # Clean up memory
                if 'analyzer' in locals():
                    del analyzer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except KeyboardInterrupt:
                print("\n⏹️  Testing interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error with {model_name}: {e}")
                self.results.append(ModelPerformance(
                    model_name=model_name,
                    works=False,
                    error_message=str(e)[:100]
                ))
        
        return self.results
    
    def create_performance_table(self) -> str:
        """Create a performance comparison table."""
        data = []
        for perf in self.results:
            data.append([
                perf.model_name.split('/')[-1],  # Short name
                "✅" if perf.works else "❌",
                f"{perf.load_time:.1f}s" if perf.works else "N/A",
                f"{perf.inference_time:.1f}s" if perf.works else "N/A",
                f"{perf.memory_usage_mb:.0f}MB" if perf.works and perf.memory_usage_mb > 0 else "N/A",
                perf.error_message[:30] + "..." if perf.error_message and len(perf.error_message) > 30 else perf.error_message
            ])
        
        headers = ["Model", "Works", "Load Time", "Inference Time", "Memory", "Error"]
        return tabulate(data, headers=headers, tablefmt="grid")
    
    def create_results_comparison_table(self, question_index: int = 0) -> str:
        """Create a table comparing answers for a specific question."""
        if question_index >= len(LicenseAnalyzerLLM(verbose=False).QUESTIONS):
            return "Invalid question index"
        
        working_results = [r for r in self.results if r.works and r.result]
        if not working_results:
            return "No working models to compare"
        
        question = working_results[0].result.answers.keys()
        question_text = list(question)[question_index]
        
        data = []
        for perf in working_results:
            answer = perf.result.answers.get(question_text, "No answer")
            # Truncate long answers
            if len(answer) > 80:
                answer = answer[:77] + "..."
            
            data.append([
                perf.model_name.split('/')[-1],
                answer
            ])
        
        headers = ["Model", f"Answer to: {question_text[:60]}..."]
        return tabulate(data, headers=headers, tablefmt="grid", maxcolwidths=[20, 80])
    
    def save_detailed_results(self, output_file: str):
        """Save detailed results to JSON file."""
        detailed_results = []
        for perf in self.results:
            result_data = {
                "model_name": perf.model_name,
                "works": perf.works,
                "load_time": perf.load_time,
                "inference_time": perf.inference_time,
                "memory_usage_mb": perf.memory_usage_mb,
                "error_message": perf.error_message,
                "answers": perf.result.answers if perf.result else {},
                "summary": perf.result.summary if perf.result else None,
                "confidence": perf.result.confidence if perf.result else None
            }
            detailed_results.append(result_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to: {output_file}")


def main():
    """Main function for interactive model comparison."""
    license_path = input("Enter path to LICENSE file: ").strip()
    
    if not Path(license_path).exists():
        print(f"❌ File not found: {license_path}")
        return
    
    print("\nChoose comparison mode:")
    print("1. Quick comparison (CPU-friendly models)")
    print("2. Full comparison (all models)")
    print("3. GPU comparison (quantized models)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    comparator = ModelComparison(use_4bit=(choice == "3"))
    
    # Adjust model list based on choice
    if choice == "1":
        # CPU-friendly models only
        comparator.MODELS_TO_TEST = [
            "microsoft/phi-2",
            "microsoft/DialoGPT-medium", 
            "microsoft/DialoGPT-small",
            "google/flan-t5-base"
        ]
    elif choice == "2":
        # Keep all models
        pass
    elif choice == "3":
        # Focus on larger, GPU-optimized models
        comparator.MODELS_TO_TEST = [
            "microsoft/Phi-3-mini-4k-instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]
    
    # Run the comparison
    results = comparator.run_comparison(license_path)
    
    # Display results
    print("\n" + "="*100)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("="*100)
    print(comparator.create_performance_table())
    
    # Show answer comparison for first question
    working_models = [r for r in results if r.works]
    if len(working_models) > 1:
        print("\n" + "="*100)
        print("🔍 ANSWER COMPARISON (First Question)")
        print("="*100)
        print(comparator.create_results_comparison_table(0))
        
        # Ask if user wants to see more comparisons
        show_more = input("\nShow detailed comparison for all questions? (y/n): ").strip().lower()
        if show_more == 'y':
            questions = list(working_models[0].result.answers.keys())
            for i, question in enumerate(questions):
                print(f"\n📋 Question {i+1}: {question}")
                print("-" * 80)
                print(comparator.create_results_comparison_table(i))
    
    # Save detailed results
    save_results = input("\nSave detailed results to JSON? (y/n): ").strip().lower()
    if save_results == 'y':
        output_file = Path(license_path).stem + "_model_comparison.json"
        comparator.save_detailed_results(output_file)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()