import os
import json
import sys
sys.path.append("****")
import pickle
from huggingface_hub import login
login(token="****")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from aisafetylab.attack.attackers.tap import TAPManager

attack_model_path = 'lmsys/vicuna-7b-v1.5'
attack_model_name = 'vicuna'

target_model_path = 'meta-llama/Llama-2-7b-hf'
target_model_name = 'Llama-2'

evaluator_model_path = 'meta-llama/Llama-Guard-3-8B'
evaluator_type='llamaguard3'

data_path = 'thu-coai/AISafetyLab_Datasets/harmbench_standard'

api_key ='****'
api_base = '****'

with open("data_export_file.pkl", "rb") as f:
    d = pickle.load(f)

i = int(sys.argv[1])
print("Argument i is =", i)

print("instance_query is == ", d[i])
pt = 'tap_check_llama_' + str(i) + '.jsonl'

tap = TAPManager(
    data_path=data_path,
    attack_model_path=attack_model_path,
    attack_model_name=attack_model_name,
    target_model_path=target_model_path,
    target_model_name=target_model_name,
    eval_model_name='openai',
    eval_model_path='gpt-4o-mini',
    openai_key=api_key,
    openai_url=api_base,
    evaluator_model_path=evaluator_model_path,
    evaluator_type=evaluator_type,
)

res = tap.mutate(d[i])

print("Response is == ", res, "\n\n\n\n\n\n")