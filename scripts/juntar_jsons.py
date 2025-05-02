import json

# Read the JSON file
with open('groq/ENtoPT.json', 'r', encoding='utf-8') as file:
    data1 = json.load(file)
with open('groq/PTtoEN.json', 'r', encoding='utf-8') as file:
    data2 = json.load(file)

# Process each object in the JSON
new_data = []
indice = 0
for item in data1:
    result = {
        "ingles_original": item["fraseEN"],
        "portugues_traduzido": item["traducaoPT"],
        "ingles_traduzido": data2[indice]["traducaoEN"]
    }

    new_data.append(result)
    result = {}
    indice += 1

# Write the new data to a new JSON file
with open('groq/frases_traduzidas.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

