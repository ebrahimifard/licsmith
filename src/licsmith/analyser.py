import json
from pathlib import Path
from typing import List
import torch


from llm_wrapper import LLMWrapper


class LicenseAnalyser:
    def __init__(self):
        self.QUESTIONS: List[str] = [
            "What is the name/type of the license (MIT, Apache 2.0, GPLv3, BSD, proprietary, etc.)?",
            "Is it a standard open-source license or a custom license?",
            "What is the copyright notice (holder(s) and year(s))?",
            "Does the license allow commercial use?",
            "Does it allow private use?",
            "Can the code be modified?",
            "Can modified versions be distributed?",
            "Is sublicensing permitted (can I re-license my modifications)?",
            "Is attribution required (do I need to credit the author)?",
            "Must I include a copy of the license in redistributions?",
            "Are there copyleft obligations (must derivative works use the same license)?",
            "Are there requirements to disclose source code of modifications?",
            "Does it impose notice requirements (e.g., stating changes made)?",
            "Does the license disclaim warranty?",
            "Does it limit or deny liability of the authors?",
            "Are there patent clauses (grant, termination, or restrictions)?",
            "Are there restrictions on field of use (e.g., non-commercial only)?",
            "Are there jurisdiction or governing law clauses?",
            "Does the license include additional permissions beyond the standard template?",
            "Does it include additional restrictions (more restrictive than normal)?",
            "Is there dual licensing (can I choose between two licenses)?",
            "Does the project have a CLA (Contributor License Agreement) that affects usage?",
            "Can I use this package in proprietary/commercial software?",
            "Do I need to make my own code open-source if I use this package?",
            "What attribution text or notices must I include in my app or docs?",
            "Do I need to track license compatibility with other dependencies?",
        ]

        self.model_name = None
        self.device = None
        self.context_reserve_tokens = None
        self.prefer_4bit = None
        self.verbose = None

        self.MAX_NEW_TOKENS = None
        self.TEMPERATURE = None
        self.TOP_P = None
        self.RANDOM_SEED = None

        self.model_load_time = None

        self.results = None

    def analyseUsingLLM(
        self, license_file_path = None,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        context_reserve_tokens: int = 1024,
        prefer_4bit: bool = True,
        verbose: bool = True):

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

        wrapper = LLMWrapper(license_file_path, self.create_config_file())
        self.results = wrapper.launch_model()

    def create_config_file(self):
        return {
            "QUESTIONS": self.QUESTIONS,
            "model_name": self.model_name,
            "device": self.device,
            "context_reserve_tokens": self.context_reserve_tokens,
            "prefer_4bit": self.prefer_4bit,
            "verbose": self.verbose,
            "MAX_NEW_TOKENS": self.MAX_NEW_TOKENS,
            "TEMPERATURE": self.TEMPERATURE,
            "TOP_P": self.TOP_P,
            "RANDOM_SEED": self.RANDOM_SEED,
            "model_load_time": self.model_load_time,
        }


    
    # def print_results(self, result: QAResult):
    #     """Print results in a nice format."""
    #     print("\n" + "="*80)
    #     print("🔍 LICENSE ANALYSIS RESULTS")
    #     print("="*80)
    #     for i, q in enumerate(self.QUESTIONS, 1):
    #         ans = result.answers.get(q, "")
    #         print(f"\n{i}. {q}")
    #         print(f"   📝 {ans}")
    #         print("-"*60)
        
    #     if result.summary:
    #         print(f"\n📌 Summary:")
    #         print(f"   {result.summary}")
        
    #     if result.confidence is not None:
    #         try:
    #             conf_val = float(result.confidence)
    #             print(f"\n🎯 Confidence: {conf_val:.2f}")
    #         except (ValueError, TypeError):
    #             print(f"\n🎯 Confidence: {result.confidence}")

    # def save_results(self, result: QAResult, license_file_path: str):
        """Save results to JSON file."""
        license_file = Path(license_file_path)
        out_path = license_file.with_suffix(".license-analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {out_path}")

    def save_results(self, saving_path):
        json.dump(open(saving_path, "w"), self.results)

    def print_results(self):
        for q,a in self.results.items():
            print(q)
            print("\n\n")
            print(a)
            print("\n")
            print("="*50)

def analyse_license(license_file_path: str, save_results: bool = True, analysis_method: str = "LLM", saving_path: str = "./"):
    """
    Simple function to analyse a license file using Qwen2.5-7B-Instruct.
    
    Args:
        license_file_path: Path to the license file
        save_results: Whether to save results to JSON file
    
    Returns:
        QAResult object with structured analysis
    """

    if analysis_method is "LLM":
        # Create analyser with Qwen2.5-7B-Instruct
        LLM_analyser = LicenseAnalyser()
        LLM_analyser.analyseUsingLLM(license_file_path) 
        LLM_analyser.print_results()

        # Save if requested
        if save_results:
            LLM_analyser.save_results(saving_path)
    else:
        print("We are working on other license analysis methods. Stay tuned ...")
        quit()
    
    return result


if __name__ == "__main__":
    license_path = input("Enter path to LICENSE file: ").strip()

    if not Path(license_path).exists():
        print(f"❌ File not found: {license_path}")
        quit()
    try:
        save_result_answer = input("Do you want to save the results? [Y/N] \n")
        save_result_flag = True if save_result_answer == "Y" else False
        path_to_save = None
        if save_result_flag:
            path_to_save = input("Enter path you like to save the results: ").strip()
        result = analyse_license(license_path, save_result_flag, "LLM", path_to_save)
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")