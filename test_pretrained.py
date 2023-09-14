from transformers import AutoTokenizer
import transformers
import torch

model = "llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(
    model,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def gen(x, max_length=200):
    sequences = pipeline(
        x,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )

    return sequences[0]["generated_text"].replace(x, "")


print('Question: please explain about runtime security project Falco.')
print(gen('please explain about runtime security project Falco.'))


print('\n\nBelow is system log which is printed by Falco. Please explain meaning of it. \n  Critical A shell was spawned in a container with an attached terminal.\n\n')
print(gen('below is system log which is printed by Falco. Please explain meaning of it. \n  Critical A shell was spawned in a container with an attached terminal.'))

