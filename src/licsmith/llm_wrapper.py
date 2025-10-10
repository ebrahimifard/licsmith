import time

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import os
import torch, gc


class LLMWrapper:
    def __init__(self, license_file_path, config_file):
        self.license_file_path = license_file_path
        self.config_file = config_file
        self.model_load_time = None


        # Cleanup at the START of the function
        if 'model' in globals():
            del model
        if 'tokenizer' in globals():
            del tokenizer
        
        torch.cuda.empty_cache()
        gc.collect()

    def launch_model(self):
        qa = {}
        start_time = time.time()
        model_name = self.config_file["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model =  AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map={"": 0}, 
        )
        self.model_load_time = time.time() - start_time


        SYSTEM_PROMPT = """You are a legal analysis assistant (not a lawyer).
        You will receive a LICENSE text and a QUESTION.
        Answer strictly using the LICENSE text. Follow this 3-line format exactly:

        First line: A short, direct answer (Yes / No / Possibly / Not found).
        Second line: empty line
        Third line: A longer explanation with brief citation (e.g., section name or quote).

        Example:
        Can I use this software commercially?

        Yes.

        Explanation: Section 2 grants permission to use and distribute for any purpose, including commercial use.

        If information is missing, say "Not found." and explain briefly why.
        """


        with open(self.license_file_path, "r", encoding="utf-8") as f:
            license_text = f.read()

        for q in self.config_file["QUESTIONS"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"LICENSE:\n{license_text}\n\nQuestion: {q}"}
            ]

            # build the formatted chat text
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # tokenize and run generation
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False
            )

            # decode the answer
            answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            qa[q] = answer 
            # print(f"\nQ: {q}\nA: {answer}\n" + "-"*80)
        return qa