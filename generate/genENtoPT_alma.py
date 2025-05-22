from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import pandas as pd
import gc
import os
import json

model = AutoModelForCausalLM.from_pretrained("haoranxu/ALMA-13B", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("haoranxu/ALMA-13B", padding_side='left')


generation_config = GenerationConfig(
    max_new_tokens=200,
    num_beams=5,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

anotacoes = []
df = pd.read_csv("comparacao_datasets/newsmet.csv", encoding='utf-8')

for frase in df["Text"]:
    prompt = f"""Translate this from English into Portuguese:
                English: {frase}
                Portuguese:"""

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            **generation_config.to_dict()
        )

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    

    result = {
        "fraseEN": frase,
        "traducaoPT": output_text.split("Portuguese:")[-1].strip()
    }
    anotacoes.append(result)

    torch.cuda.empty_cache()
    gc.collect()

    with open("dataset_newsmet/alma/ENtoPT2.json", "w", encoding="utf-8") as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=4)




