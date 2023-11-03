from datasets import load_dataset
import pdb 

data = load_dataset("json", data_files = "data/rule.json", field = 'events')

# 질문과 답변을 모두 포함하여 간단한 학습용 프롬프트를 작성해줍니다.
data = data.map(
    lambda x: {'text': f"Based on the system information, alert the anomaly behavior.\n##system information:occured system call is {x['system_call']}, process name is {x['process_name']}, UID is {x['UID']}.\n##anomaly behavior:{x['output']} by UID {x['UID']}." if x['open_file_name'] is None else \
        f"Based on the system information, alert the anomaly behavior.\n##system information:occured system call is {x['system_call']}, process name is {x['process_name']}, open file name is {x['open_file_name']}.\n##anomaly behavior:{x['output']}, open file name is {x['open_file_name']}."
        }
)

# pretrain 모델 load
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "llama-2-13b-hf"  # huggingface에서 pretrained 모델 가져오기 
bnb_config = BitsAndBytesConfig(          # 4 bit 양자화 사용을 위한 코드 
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)


# 학습에서 peft 방식을 사용하기 위한 코드 
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# peft 방식을 사용했을 때, update하는 parameter 개수를 확인하기 위한 함수 
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# peft 방식 중에서 Lora 를 사용하기 위한 코드 
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                # r 값으로 얼마만큼의 parameter 들을 학습할 것인 지 조절
    lora_alpha=32, 
    target_modules=[
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


#실제 train 하는 부분
import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

#huggingface에서 제공하는 Trainer 모듈을 사용하여 hyperparamer 만 넣어주면 쉽게 학습 가능  
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=100, #총 66개의 데이터 셋, 배치 사이즈 2 이므로 33번에 1epoch. 
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",  
        optim="paged_adamw_8bit",
        save_steps = 10
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.eval()
model.config.use_cache = True  # silence the warnings. Please re-enable for inference!



#학습 이후 가볍게 validation 해보기 위한 함수 
def gen(system_info):
    gened = model.generate(
        **tokenizer(
            f"Based on the system information, alert the anomaly behavior.\n##system information:{system_info}\n##anomaly behavior:", 
            return_tensors='pt', 
            return_token_type_ids=False
        ), 
        max_new_tokens=30,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    result = tokenizer.decode(gened[0])
    start_index = result.find("##anomaly")
    end_index = start_index + result[start_index:].find(".")
    print(result[:end_index+1])


gen("Occured system call is execveat, process name is bash, UID is 1000.")
gen("Occured system call is openat2, process name is a.out, open file name is /etc/file1.")

