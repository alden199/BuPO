from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/ssd/tanyuqiao/models/Qwen2.5-Math-1.5B", dtype="auto", device_map="auto")
import pdb; pdb.set_trace()