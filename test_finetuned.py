# instruction tuning을 완료한 모델 test를 위한 코드 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


#peft 방식으로 fine tuning 한 모델 불러오기  
peft_model_id = "./outputs/checkpoint-1"  #finetuned 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(    #test할때도 동일하게 4비트 양자화 사용 
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()


#학습 이후 가볍게 validation 해보기 위한 함수 
def gen(system_info):
    gened = model.generate(
        **tokenizer(
            f"Based on the system information, alert the anomaly behavior.##system information:{system_info}##anomaly behavior:", 
            return_tensors='pt', 
            return_token_type_ids=False
        ), 
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))


gen("Occured system call is execve2, process name is bash, UID is 1000.")
gen("Occured system call is openat2, process name is a.out, open file name is /etc/file1.")


