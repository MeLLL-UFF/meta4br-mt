import os
import json
import time
import concurrent.futures
from dotenv import load_dotenv
from datasets import load_dataset

# -----------------------
# 1. IMPORT AND CONFIGURE GEMINI
# -----------------------
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_PROJECT = os.environ['GOOGLE_PROJECT']

# Create the Gemini client
client = genai.Client(
    vertexai=True,
    project=GOOGLE_PROJECT,
    location="us-central1",
)

# -----------------------
# 2. SET PARAMETERS
# -----------------------
MODEL_NAME = "gemini-2.0-flash-001"
CSV_FILEPATH = "news_reports_dataset.csv"

NEWS_COLUMN_NAME = "news"
REPORTS_COLUMN_NAME = "reports"

INTERMEDIATE_JSON_PATH = "5_gemini_intermediate_results.json" #CHANGE PATHS!!!
FINAL_JSON_PATH = "5_gemini_final_results.json"

START_INDEX = 8001 #CHANGE INDEX!!!
END_INDEX = 10706
BATCH_SIZE = 10
INTERMEDIATE_SAVE_STEP = 10

# Translation parameters
SOURCE_LANG = "Chinese"
TARGET_LANG = "Brazilian Portuguese"

# API call parameters
MAX_TOKENS = 2000
TEMPERATURE = 0.8
SLEEP_TIME = 15 #Menos que isso começam a vir erros 429 (too many requests)

# -----------------------------------------
# 3. LOAD YOUR CSV DATASET
# -----------------------------------------
dataset = load_dataset("csv", data_files=CSV_FILEPATH)
data = dataset["train"]
data = data.remove_columns(
    [col for col in data.column_names if col not in [NEWS_COLUMN_NAME, REPORTS_COLUMN_NAME]]
)

if END_INDEX is not None:
    data = data.select(range(START_INDEX, END_INDEX + 1))
else:
    data = data.select(range(START_INDEX, len(data)))

# --------------------------------------------------
# 4. FUNCTION TO CALL GEMINI AND FETCH TRANSLATIONS
# --------------------------------------------------
def fetch_translation(message_pair):
    """
    Sends a translation request to Gemini.

    message_pair is a list of exactly two strings:
      [system_text, user_text]
    
    Returns the translated text or an error message.
    """
    try:
        time.sleep(SLEEP_TIME)
        # Unpack the system and user text
        system_text, user_text = message_pair

        # Build the config with system instruction
        generate_content_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=0.95,
            max_output_tokens=MAX_TOKENS,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            # Pass the system text here
            system_instruction=[types.Part.from_text(text=system_text)],
        )

        # Build the user content
        contents = [
            user_text
        ]

        # Make the API call
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        )

        translated_text = response.text

    except Exception as e:
        translated_text = f"Error: {str(e)}"

    return translated_text


def generate_batch_translations(batch_texts, source_lang, target_lang):
    """
    For each text in the batch, build the system+user message pair, 
    then fetch translations concurrently.
    """
    # The "system" message
    system_msg = (
        "You are an assistant that only provides translations. "
        "When the user inputs text in one language, you respond with the "
        "translated version in the target language WITHOUT any additional commentary, "
        "explanations, or formatting."
    )

    # Build message pairs
    message_pairs = []
    for text in batch_texts:
        user_msg = (
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"Be as precise as possible in retaining the information conveyed.\n\n{text}\n"
        )
        message_pairs.append([system_msg, user_msg])

    # Make concurrent requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        translations = list(executor.map(fetch_translation, message_pairs))

    return translations

# -----------------------------------------------------------
# 5. ITERATE OVER THE DATASET IN BATCHES AND BUILD RESULTS
# -----------------------------------------------------------
results = []
num_rows = len(data)

for start_i in range(0, num_rows, BATCH_SIZE):
    end_i = min(start_i + BATCH_SIZE, num_rows)
    batch = data.select(range(start_i, end_i))

    news_texts = batch[NEWS_COLUMN_NAME]
    reports_texts = batch[REPORTS_COLUMN_NAME]

    # 1) Translate the news column 
    translated_news_list = generate_batch_translations(news_texts, SOURCE_LANG, TARGET_LANG)
    
    # 2) Translate the reports column
    translated_reports_list = generate_batch_translations(reports_texts, SOURCE_LANG, TARGET_LANG)
    
    # Combine results for each item in the batch
    for idx_in_batch, _ in enumerate(batch):
        absolute_idx = START_INDEX + start_i + idx_in_batch
        if idx_in_batch < len(translated_news_list) and idx_in_batch < len(translated_reports_list):
            translated_dict = {
                "id": absolute_idx,
                "news_translation": translated_news_list[idx_in_batch],
                "reports_translation": translated_reports_list[idx_in_batch],
            }
            results.append(translated_dict)
    
    # Save intermediate results at intervals
    if start_i % INTERMEDIATE_SAVE_STEP == 0:
        with open(INTERMEDIATE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved intermediate results up to index {absolute_idx}")

# -------------------------------------
# 6. SAVE THE FINAL TRANSLATION OUTPUT
# -------------------------------------
with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Translation complete! Final results saved to {FINAL_JSON_PATH}")
if (num_rows + START_INDEX) >= INTERMEDIATE_SAVE_STEP:
    print(f"Intermediate results saved to {INTERMEDIATE_JSON_PATH}")
