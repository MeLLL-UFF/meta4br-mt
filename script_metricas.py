# pip install evaluate 
# pip install rouge-score 
# pip install absl-py 
# pip install nltk 
# pip install bert_score 
# pip install git+https://github.com/google-research/bleurt.git
# pip install unbabel-comet

import evaluate
import json

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
bleurt = evaluate.load('bleurt', 'bleurt-large-512')
comet = evaluate.load('comet')

with open('gpt/frases_traduzidas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

vetor = [
    {
        "ingles_original": "word becomes a rock",
        "portugues_traduzido": "a palavra se torna uma rocha",
        "ingles_traduzido": "\"The word becomes a rock.\"",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.345720784641941,
            "precisions": [
                0.5,
                0.42857142857142855,
                0.3333333333333333,
                0.2
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8448767066001892
            ],
            "recall": [
                0.9390988945960999
            ],
            "f1": [
                0.8894995450973511
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7902722358703613
            ]
        },
        "COMET": {
            "scores": [
                0.9271788001060486
            ]
        }
    },
    {
        "ingles_original": "silence becomes a thunder",
        "portugues_traduzido": "o silêncio se torna um trovão",
        "ingles_traduzido": "Silence becomes a thunder.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9539317488670349
            ],
            "recall": [
                0.971666157245636
            ],
            "f1": [
                0.9627172946929932
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0370476245880127
            ]
        },
        "COMET": {
            "scores": [
                0.9469823241233826
            ]
        }
    },
    {
        "ingles_original": "the doctor walks to the school.",
        "portugues_traduzido": "O médico vai à escola.",
        "ingles_traduzido": "The doctor goes to school.",
        "ROUGE": {
            "rouge1": 0.7272727272727272,
            "rouge2": 0.22222222222222224,
            "rougeL": 0.7272727272727272,
            "rougeLsum": 0.7272727272727272
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6666666666666666,
                0.2,
                0.0,
                0.0
            ],
            "brevity_penalty": 0.846481724890614,
            "length_ratio": 0.8571428571428571,
            "translation_lenght": 6,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                0.911136269569397
            ],
            "recall": [
                0.8975892066955566
            ],
            "f1": [
                0.904312014579773
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8030862808227539
            ]
        },
        "COMET": {
            "scores": [
                0.7714747786521912
            ]
        }
    },
    {
        "ingles_original": "strength becomes a diamond",
        "portugues_traduzido": "a força se torna um diamante",
        "ingles_traduzido": "\"strength becomes a diamond\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8651002049446106
            ],
            "recall": [
                0.9687052369117737
            ],
            "f1": [
                0.9139760136604309
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7664586305618286
            ]
        },
        "COMET": {
            "scores": [
                0.9004654288291931
            ]
        }
    },
    {
        "ingles_original": "time has a mountain",
        "portugues_traduzido": "o tempo tem uma montanha",
        "ingles_traduzido": "Time has a mountain.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9115222692489624
            ],
            "recall": [
                0.9774556159973145
            ],
            "f1": [
                0.9433382749557495
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9630624651908875
            ]
        },
        "COMET": {
            "scores": [
                0.894554078578949
            ]
        }
    },
    {
        "ingles_original": "mind becomes a gold",
        "portugues_traduzido": "mente se torna ouro",
        "ingles_traduzido": "mind becomes gold",
        "ROUGE": {
            "rouge1": 0.8571428571428571,
            "rouge2": 0.4,
            "rougeL": 0.8571428571428571,
            "rougeLsum": 0.8571428571428571
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                1.0,
                0.5,
                0.0,
                0.0
            ],
            "brevity_penalty": 0.7165313105737893,
            "length_ratio": 0.75,
            "translation_lenght": 3,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9828553795814514
            ],
            "recall": [
                0.9398723244667053
            ],
            "f1": [
                0.960883378982544
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7682293057441711
            ]
        },
        "COMET": {
            "scores": [
                0.9516282677650452
            ]
        }
    },
    {
        "ingles_original": "word has a rock",
        "portugues_traduzido": "A palavra tem uma pedra.",
        "ingles_traduzido": "The word has a stone.",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.5714285714285715,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.5,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8477987051010132
            ],
            "recall": [
                0.8775043487548828
            ],
            "f1": [
                0.8623957633972168
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.03087671846151352
            ]
        },
        "COMET": {
            "scores": [
                0.7778559327125549
            ]
        }
    },
    {
        "ingles_original": "dream has a star",
        "portugues_traduzido": "o sonho tem uma estrela",
        "ingles_traduzido": "\"The dream has a star.\"",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.345720784641941,
            "precisions": [
                0.5,
                0.42857142857142855,
                0.3333333333333333,
                0.2
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8352375626564026
            ],
            "recall": [
                0.9416993260383606
            ],
            "f1": [
                0.8852792382240295
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.3745211064815521
            ]
        },
        "COMET": {
            "scores": [
                0.9164593815803528
            ]
        }
    },
    {
        "ingles_original": "passion is a light",
        "portugues_traduzido": "A paixão é uma luz.",
        "ingles_traduzido": "\"Passion is a light.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8599574565887451
            ],
            "recall": [
                0.9632601737976074
            ],
            "f1": [
                0.9086822867393494
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8567508459091187
            ]
        },
        "COMET": {
            "scores": [
                0.8317474722862244
            ]
        }
    },
    {
        "ingles_original": "dream has a star",
        "portugues_traduzido": "o sonho tem uma estrela",
        "ingles_traduzido": "The dream has a star.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8622972965240479
            ],
            "recall": [
                0.9409378170967102
            ],
            "f1": [
                0.8999027609825134
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.37839314341545105
            ]
        },
        "COMET": {
            "scores": [
                0.9425446391105652
            ]
        }
    },
    {
        "ingles_original": "courage has a sun",
        "portugues_traduzido": "A coragem tem um sol.",
        "ingles_traduzido": "\"Courage has a sun.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8626562356948853
            ],
            "recall": [
                0.9582747220993042
            ],
            "f1": [
                0.9079549312591553
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8209140300750732
            ]
        },
        "COMET": {
            "scores": [
                0.8216587901115417
            ]
        }
    },
    {
        "ingles_original": "passion has a light",
        "portugues_traduzido": "a paixão tem uma luz",
        "ingles_traduzido": "Passion has a light.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9037472009658813
            ],
            "recall": [
                0.9729728698730469
            ],
            "f1": [
                0.9370833039283752
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0126057863235474
            ]
        },
        "COMET": {
            "scores": [
                0.9408701062202454
            ]
        }
    },
    {
        "ingles_original": "dream has a star",
        "portugues_traduzido": "sonho tem uma estrela",
        "ingles_traduzido": "Dream has a star.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9149425625801086
            ],
            "recall": [
                0.9732040762901306
            ],
            "f1": [
                0.9431744813919067
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9997314214706421
            ]
        },
        "COMET": {
            "scores": [
                0.9345136284828186
            ]
        }
    },
    {
        "ingles_original": "the driver visits the cafe.",
        "portugues_traduzido": "o motorista visita o café.",
        "ingles_traduzido": "The driver visits the café.",
        "ROUGE": {
            "rouge1": 0.8000000000000002,
            "rouge2": 0.75,
            "rougeL": 0.8000000000000002,
            "rougeLsum": 0.8000000000000002
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6666666666666666,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.0
            ],
            "recall": [
                1.0
            ],
            "f1": [
                1.0
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0576101541519165
            ]
        },
        "COMET": {
            "scores": [
                0.9417674541473389
            ]
        }
    },
    {
        "ingles_original": "dream is a star",
        "portugues_traduzido": "sonho é uma estrela",
        "ingles_traduzido": "Dream is a star.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9126494526863098
            ],
            "recall": [
                0.9738128781318665
            ],
            "f1": [
                0.9422396421432495
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9557849168777466
            ]
        },
        "COMET": {
            "scores": [
                0.9217413067817688
            ]
        }
    },
    {
        "ingles_original": "spirit is a storm",
        "portugues_traduzido": "o espírito é uma tempestade",
        "ingles_traduzido": "The spirit is a storm.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8924791216850281
            ],
            "recall": [
                0.9518487453460693
            ],
            "f1": [
                0.9212083220481873
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9107125997543335
            ]
        },
        "COMET": {
            "scores": [
                0.9010183811187744
            ]
        }
    },
    {
        "ingles_original": "word is a rock",
        "portugues_traduzido": "A palavra é uma rocha.",
        "ingles_traduzido": "The word is a rock.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8598257303237915
            ],
            "recall": [
                0.8922830820083618
            ],
            "f1": [
                0.8757538199424744
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8364371657371521
            ]
        },
        "COMET": {
            "scores": [
                0.8961144089698792
            ]
        }
    },
    {
        "ingles_original": "smile becomes a wind",
        "portugues_traduzido": "sorriso se torna um vento",
        "ingles_traduzido": "\"smile becomes a wind\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8788187503814697
            ],
            "recall": [
                0.9692061543464661
            ],
            "f1": [
                0.9218020439147949
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8746273517608643
            ]
        },
        "COMET": {
            "scores": [
                0.9110600352287292
            ]
        }
    },
    {
        "ingles_original": "mind has a gold",
        "portugues_traduzido": "A mente tem um ouro.",
        "ingles_traduzido": "The mind has a gold.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8623405694961548
            ],
            "recall": [
                0.9413684606552124
            ],
            "f1": [
                0.9001232385635376
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8359895348548889
            ]
        },
        "COMET": {
            "scores": [
                0.8982425928115845
            ]
        }
    },
    {
        "ingles_original": "hope becomes a ice",
        "portugues_traduzido": "a esperança se torna um gelo",
        "ingles_traduzido": "\"Hope turns into ice.\"",
        "ROUGE": {
            "rouge1": 0.5,
            "rouge2": 0.0,
            "rougeL": 0.5,
            "rougeLsum": 0.5
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.14285714285714285,
                0.0,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8017934560775757
            ],
            "recall": [
                0.8562150001525879
            ],
            "f1": [
                0.8281110525131226
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.27565574645996094
            ]
        },
        "COMET": {
            "scores": [
                0.8687917590141296
            ]
        }
    },
    {
        "ingles_original": "love becomes a moon",
        "portugues_traduzido": "o amor se torna uma lua",
        "ingles_traduzido": "Love becomes a moon.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8955600261688232
            ],
            "recall": [
                0.9667359590530396
            ],
            "f1": [
                0.9297878742218018
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.952788233757019
            ]
        },
        "COMET": {
            "scores": [
                0.9515643119812012
            ]
        }
    },
    {
        "ingles_original": "the artist works in the library.",
        "portugues_traduzido": "o artista trabalha na biblioteca.",
        "ingles_traduzido": "The artist works in the library.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.8091067115702212,
            "precisions": [
                0.8571428571428571,
                0.8333333333333334,
                0.8,
                0.75
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 7,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0512031316757202
            ]
        },
        "COMET": {
            "scores": [
                0.9326633214950562
            ]
        }
    },
    {
        "ingles_original": "the doctor walks to the school.",
        "portugues_traduzido": "O médico caminha para a escola.",
        "ingles_traduzido": "The doctor walks to the school.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.8091067115702212,
            "precisions": [
                0.8571428571428571,
                0.8333333333333334,
                0.8,
                0.75
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 7,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0738896131515503
            ]
        },
        "COMET": {
            "scores": [
                0.9442793726921082
            ]
        }
    },
    {
        "ingles_original": "the chef writes the restaurant.",
        "portugues_traduzido": "o chefe escreve o restaurante.",
        "ingles_traduzido": "The chef writes the restaurant.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.000000238418579
            ],
            "recall": [
                1.000000238418579
            ],
            "f1": [
                1.000000238418579
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0537066459655762
            ]
        },
        "COMET": {
            "scores": [
                0.9305756092071533
            ]
        }
    },
    {
        "ingles_original": "memory becomes a ocean",
        "portugues_traduzido": "A memória se torna um oceano.",
        "ingles_traduzido": "Memory becomes an ocean.",
        "ROUGE": {
            "rouge1": 0.75,
            "rouge2": 0.3333333333333333,
            "rougeL": 0.75,
            "rougeLsum": 0.75
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.4,
                0.0,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9013225436210632
            ],
            "recall": [
                0.9581414461135864
            ],
            "f1": [
                0.9288639426231384
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9381914734840393
            ]
        },
        "COMET": {
            "scores": [
                0.9573150873184204
            ]
        }
    },
    {
        "ingles_original": "the child drinks the market.",
        "portugues_traduzido": "A criança bebe o mercado.",
        "ingles_traduzido": "The child drinks the market.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.0
            ],
            "recall": [
                1.0
            ],
            "f1": [
                1.0
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.05592679977417
            ]
        },
        "COMET": {
            "scores": [
                0.9117934107780457
            ]
        }
    },
    {
        "ingles_original": "smile is a wind",
        "portugues_traduzido": "sorriso é um vento",
        "ingles_traduzido": "\"Smile is a wind.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8556344509124756
            ],
            "recall": [
                0.9467790126800537
            ],
            "f1": [
                0.8989022374153137
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.852859377861023
            ]
        },
        "COMET": {
            "scores": [
                0.8402067422866821
            ]
        }
    },
    {
        "ingles_original": "courage is a sun",
        "portugues_traduzido": "A coragem é um sol.",
        "ingles_traduzido": "Courage is a sun.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9075623154640198
            ],
            "recall": [
                0.9656181335449219
            ],
            "f1": [
                0.9356905817985535
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9203808307647705
            ]
        },
        "COMET": {
            "scores": [
                0.8627174496650696
            ]
        }
    },
    {
        "ingles_original": "the chef writes the restaurant.",
        "portugues_traduzido": "o chef escreve o restaurante.",
        "ingles_traduzido": "The chef writes the restaurant.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.000000238418579
            ],
            "recall": [
                1.000000238418579
            ],
            "f1": [
                1.000000238418579
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0537066459655762
            ]
        },
        "COMET": {
            "scores": [
                0.9364845156669617
            ]
        }
    },
    {
        "ingles_original": "strength becomes a diamond",
        "portugues_traduzido": "a força se torna um diamante",
        "ingles_traduzido": "\"Strength becomes a diamond.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8620654940605164
            ],
            "recall": [
                0.9632476568222046
            ],
            "f1": [
                0.9098522067070007
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7748542428016663
            ]
        },
        "COMET": {
            "scores": [
                0.8913112878799438
            ]
        }
    },
    {
        "ingles_original": "the doctor walks to the school.",
        "portugues_traduzido": "O médico caminha para a escola.",
        "ingles_traduzido": "The doctor walks to the school.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.8091067115702212,
            "precisions": [
                0.8571428571428571,
                0.8333333333333334,
                0.8,
                0.75
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 7,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0738896131515503
            ]
        },
        "COMET": {
            "scores": [
                0.9442793726921082
            ]
        }
    },
    {
        "ingles_original": "heart has a tree",
        "portugues_traduzido": "o coração tem uma árvore",
        "ingles_traduzido": "the heart has a tree",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.668740304976422,
            "precisions": [
                0.8,
                0.75,
                0.6666666666666666,
                0.5
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9369064569473267
            ],
            "recall": [
                0.9743838310241699
            ],
            "f1": [
                0.9552777409553528
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9492064714431763
            ]
        },
        "COMET": {
            "scores": [
                0.9356436729431152
            ]
        }
    },
    {
        "ingles_original": "love has a moon",
        "portugues_traduzido": "o amor tem uma lua",
        "ingles_traduzido": "Love has a moon.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8988685607910156
            ],
            "recall": [
                0.9663658142089844
            ],
            "f1": [
                0.9313958883285522
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0126736164093018
            ]
        },
        "COMET": {
            "scores": [
                0.9600141644477844
            ]
        }
    },
    {
        "ingles_original": "strength becomes a diamond",
        "portugues_traduzido": "força se torna um diamante",
        "ingles_traduzido": "strength becomes a diamond",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 1.0,
            "precisions": [
                1.0,
                1.0,
                1.0,
                1.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 4,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9999999403953552
            ],
            "recall": [
                0.9999999403953552
            ],
            "f1": [
                0.9999999403953552
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0345021486282349
            ]
        },
        "COMET": {
            "scores": [
                0.9878804087638855
            ]
        }
    },
    {
        "ingles_original": "smile is a wind",
        "portugues_traduzido": "sorriso é um vento",
        "ingles_traduzido": "A smile is a wind.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.880831241607666
            ],
            "recall": [
                0.9295331239700317
            ],
            "f1": [
                0.9045271277427673
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.6455758810043335
            ]
        },
        "COMET": {
            "scores": [
                0.9184402823448181
            ]
        }
    },
    {
        "ingles_original": "eye is a flame",
        "portugues_traduzido": "\"olho é uma chama\"",
        "ingles_traduzido": "\"the eye is a flame\"",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.4111336169005197,
            "precisions": [
                0.5714285714285714,
                0.5,
                0.4,
                0.25
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8480238318443298
            ],
            "recall": [
                0.9565656781196594
            ],
            "f1": [
                0.8990305066108704
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.6487215757369995
            ]
        },
        "COMET": {
            "scores": [
                0.9001632928848267
            ]
        }
    },
    {
        "ingles_original": "smile is a wind",
        "portugues_traduzido": "sorriso é um vento",
        "ingles_traduzido": "\"Smile is a wind.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8556344509124756
            ],
            "recall": [
                0.9467790126800537
            ],
            "f1": [
                0.8989022374153137
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.852859377861023
            ]
        },
        "COMET": {
            "scores": [
                0.8402067422866821
            ]
        }
    },
    {
        "ingles_original": "the doctor walks to the school.",
        "portugues_traduzido": "O médico vai para a escola.",
        "ingles_traduzido": "The doctor goes to school.",
        "ROUGE": {
            "rouge1": 0.7272727272727272,
            "rouge2": 0.22222222222222224,
            "rougeL": 0.7272727272727272,
            "rougeLsum": 0.7272727272727272
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6666666666666666,
                0.2,
                0.0,
                0.0
            ],
            "brevity_penalty": 0.846481724890614,
            "length_ratio": 0.8571428571428571,
            "translation_lenght": 6,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                0.911136269569397
            ],
            "recall": [
                0.8975892066955566
            ],
            "f1": [
                0.904312014579773
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8030862808227539
            ]
        },
        "COMET": {
            "scores": [
                0.7734521627426147
            ]
        }
    },
    {
        "ingles_original": "time is a mountain",
        "portugues_traduzido": "o tempo é uma montanha",
        "ingles_traduzido": "Time is a mountain.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9236505031585693
            ],
            "recall": [
                0.9790476560592651
            ],
            "f1": [
                0.950542688369751
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.939275324344635
            ]
        },
        "COMET": {
            "scores": [
                0.9523191452026367
            ]
        }
    },
    {
        "ingles_original": "courage becomes a sun",
        "portugues_traduzido": "a coragem se torna um sol",
        "ingles_traduzido": "Courage becomes a sun.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9152753353118896
            ],
            "recall": [
                0.9761828184127808
            ],
            "f1": [
                0.9447484612464905
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.97902911901474
            ]
        },
        "COMET": {
            "scores": [
                0.9076923727989197
            ]
        }
    },
    {
        "ingles_original": "mind has a gold",
        "portugues_traduzido": "A mente tem um ouro.",
        "ingles_traduzido": "The mind has gold.",
        "ROUGE": {
            "rouge1": 0.75,
            "rouge2": 0.3333333333333333,
            "rougeL": 0.75,
            "rougeLsum": 0.75
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.25,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8435726165771484
            ],
            "recall": [
                0.8917323350906372
            ],
            "f1": [
                0.8669841885566711
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.6716514825820923
            ]
        },
        "COMET": {
            "scores": [
                0.9116184115409851
            ]
        }
    },
    {
        "ingles_original": "dream becomes a star",
        "portugues_traduzido": "sonho se torna uma estrela",
        "ingles_traduzido": "dream becomes a star",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 1.0,
            "precisions": [
                1.0,
                1.0,
                1.0,
                1.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 4,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0136955976486206
            ]
        },
        "COMET": {
            "scores": [
                0.9877099990844727
            ]
        }
    },
    {
        "ingles_original": "the chef writes the restaurant.",
        "portugues_traduzido": "o chef escreve o restaurante.",
        "ingles_traduzido": "The chef writes the restaurant.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.000000238418579
            ],
            "recall": [
                1.000000238418579
            ],
            "f1": [
                1.000000238418579
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0537066459655762
            ]
        },
        "COMET": {
            "scores": [
                0.9364845156669617
            ]
        }
    },
    {
        "ingles_original": "he walks to the park.",
        "portugues_traduzido": "ele caminha para o parque.",
        "ingles_traduzido": "He walks to the park.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                0.9999997615814209
            ],
            "recall": [
                0.9999997615814209
            ],
            "f1": [
                0.9999997615814209
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0789026021957397
            ]
        },
        "COMET": {
            "scores": [
                0.9645488262176514
            ]
        }
    },
    {
        "ingles_original": "mind is a gold",
        "portugues_traduzido": "mente é ouro",
        "ingles_traduzido": "Mind is gold.",
        "ROUGE": {
            "rouge1": 0.8571428571428571,
            "rouge2": 0.4,
            "rougeL": 0.8571428571428571,
            "rougeLsum": 0.8571428571428571
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.5,
                0.0,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 4,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8808150291442871
            ],
            "recall": [
                0.9183257818222046
            ],
            "f1": [
                0.8991793394088745
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7177615761756897
            ]
        },
        "COMET": {
            "scores": [
                0.9102159142494202
            ]
        }
    },
    {
        "ingles_original": "love becomes a moon",
        "portugues_traduzido": "o amor se torna uma lua",
        "ingles_traduzido": "Love becomes a moon.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8955600261688232
            ],
            "recall": [
                0.9667359590530396
            ],
            "f1": [
                0.9297878742218018
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.952788233757019
            ]
        },
        "COMET": {
            "scores": [
                0.9515643119812012
            ]
        }
    },
    {
        "ingles_original": "eye becomes a flame",
        "portugues_traduzido": "o olho se torna uma chama",
        "ingles_traduzido": "the eye becomes a flame",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.668740304976422,
            "precisions": [
                0.8,
                0.75,
                0.6666666666666666,
                0.5
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9390923976898193
            ],
            "recall": [
                0.9754592776298523
            ],
            "f1": [
                0.95693039894104
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9714518189430237
            ]
        },
        "COMET": {
            "scores": [
                0.9543268084526062
            ]
        }
    },
    {
        "ingles_original": "mind becomes a gold",
        "portugues_traduzido": "mente se torna ouro",
        "ingles_traduzido": "mind becomes gold",
        "ROUGE": {
            "rouge1": 0.8571428571428571,
            "rouge2": 0.4,
            "rougeL": 0.8571428571428571,
            "rougeLsum": 0.8571428571428571
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                1.0,
                0.5,
                0.0,
                0.0
            ],
            "brevity_penalty": 0.7165313105737893,
            "length_ratio": 0.75,
            "translation_lenght": 3,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9828553795814514
            ],
            "recall": [
                0.9398723244667053
            ],
            "f1": [
                0.960883378982544
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7682293057441711
            ]
        },
        "COMET": {
            "scores": [
                0.9516282677650452
            ]
        }
    },
    {
        "ingles_original": "heart is a tree",
        "portugues_traduzido": "coração é uma árvore",
        "ingles_traduzido": "The heart is a tree.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8516923785209656
            ],
            "recall": [
                0.9445767998695374
            ],
            "f1": [
                0.8957330584526062
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8557640314102173
            ]
        },
        "COMET": {
            "scores": [
                0.9255202412605286
            ]
        }
    },
    {
        "ingles_original": "word has a rock",
        "portugues_traduzido": "a palavra tem uma pedra",
        "ingles_traduzido": "the word has a stone",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.5714285714285715,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.87518310546875
            ],
            "recall": [
                0.9058449268341064
            ],
            "f1": [
                0.8902500867843628
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.0549517422914505
            ]
        },
        "COMET": {
            "scores": [
                0.8283203840255737
            ]
        }
    },
    {
        "ingles_original": "soul becomes a shadow",
        "portugues_traduzido": "a alma se torna uma sombra",
        "ingles_traduzido": "the soul becomes a shadow",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.668740304976422,
            "precisions": [
                0.8,
                0.75,
                0.6666666666666666,
                0.5
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9462040066719055
            ],
            "recall": [
                0.980772078037262
            ],
            "f1": [
                0.9631779789924622
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.033528208732605
            ]
        },
        "COMET": {
            "scores": [
                0.9568583369255066
            ]
        }
    },
    {
        "ingles_original": "memory becomes a ocean",
        "portugues_traduzido": "a memória se torna um oceano",
        "ingles_traduzido": "\"Memory becomes an ocean.\"",
        "ROUGE": {
            "rouge1": 0.75,
            "rouge2": 0.3333333333333333,
            "rougeL": 0.75,
            "rougeLsum": 0.75
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.2857142857142857,
                0.0,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8505992293357849
            ],
            "recall": [
                0.9503583312034607
            ],
            "f1": [
                0.8977158069610596
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8001035451889038
            ]
        },
        "COMET": {
            "scores": [
                0.9193527102470398
            ]
        }
    },
    {
        "ingles_original": "time becomes a mountain",
        "portugues_traduzido": "o tempo se torna uma montanha",
        "ingles_traduzido": "Time becomes a mountain.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9028204679489136
            ],
            "recall": [
                0.981020987033844
            ],
            "f1": [
                0.940297544002533
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9861343502998352
            ]
        },
        "COMET": {
            "scores": [
                0.9653979539871216
            ]
        }
    },
    {
        "ingles_original": "the doctor walks to the school.",
        "portugues_traduzido": "O médico caminha para a escola.",
        "ingles_traduzido": "The doctor walks to the school.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.8091067115702212,
            "precisions": [
                0.8571428571428571,
                0.8333333333333334,
                0.8,
                0.75
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 7,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0738896131515503
            ]
        },
        "COMET": {
            "scores": [
                0.9442793726921082
            ]
        }
    },
    {
        "ingles_original": "she writes the office.",
        "portugues_traduzido": "Ela escreve o escritório.",
        "ingles_traduzido": "She writes the office.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.668740304976422,
            "precisions": [
                0.8,
                0.75,
                0.6666666666666666,
                0.5
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 5,
            "reference_lenght": 5
        },
        "BERTSCORE": {
            "precision": [
                0.9999999403953552
            ],
            "recall": [
                0.9999999403953552
            ],
            "f1": [
                0.9999999403953552
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.040927767753601
            ]
        },
        "COMET": {
            "scores": [
                0.9256632924079895
            ]
        }
    },
    {
        "ingles_original": "wisdom becomes a garden",
        "portugues_traduzido": "A sabedoria se torna um jardim.",
        "ingles_traduzido": "Wisdom becomes a garden.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9130404591560364
            ],
            "recall": [
                0.9768903255462646
            ],
            "f1": [
                0.9438868761062622
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9849439263343811
            ]
        },
        "COMET": {
            "scores": [
                0.9324638843536377
            ]
        }
    },
    {
        "ingles_original": "dream becomes a star",
        "portugues_traduzido": "sonho se torna uma estrela",
        "ingles_traduzido": "dream becomes a star",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 1.0,
            "precisions": [
                1.0,
                1.0,
                1.0,
                1.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 4,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0136955976486206
            ]
        },
        "COMET": {
            "scores": [
                0.9877099990844727
            ]
        }
    },
    {
        "ingles_original": "soul is a shadow",
        "portugues_traduzido": "A alma é uma sombra.",
        "ingles_traduzido": "The soul is a shadow.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8758991956710815
            ],
            "recall": [
                0.9436640739440918
            ],
            "f1": [
                0.9085197448730469
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9863088130950928
            ]
        },
        "COMET": {
            "scores": [
                0.9470430612564087
            ]
        }
    },
    {
        "ingles_original": "strength becomes a diamond",
        "portugues_traduzido": "a força se torna um diamante",
        "ingles_traduzido": "The force becomes a diamond.",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.5714285714285715,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.5,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8945057392120361
            ],
            "recall": [
                0.9265924096107483
            ],
            "f1": [
                0.9102663993835449
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.24911050498485565
            ]
        },
        "COMET": {
            "scores": [
                0.8726905584335327
            ]
        }
    },
    {
        "ingles_original": "courage is a sun",
        "portugues_traduzido": "A coragem é um sol.",
        "ingles_traduzido": "\"Courage is a sun.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8581953048706055
            ],
            "recall": [
                0.9552040100097656
            ],
            "f1": [
                0.9041049480438232
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7420697212219238
            ]
        },
        "COMET": {
            "scores": [
                0.8155247569084167
            ]
        }
    },
    {
        "ingles_original": "wisdom has a garden",
        "portugues_traduzido": "A sabedoria tem um jardim.",
        "ingles_traduzido": "Wisdom has a garden.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9052119851112366
            ],
            "recall": [
                0.9658809900283813
            ],
            "f1": [
                0.9345629215240479
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9994896650314331
            ]
        },
        "COMET": {
            "scores": [
                0.9222677946090698
            ]
        }
    },
    {
        "ingles_original": "wisdom becomes a garden",
        "portugues_traduzido": "a sabedoria se torna um jardim",
        "ingles_traduzido": "Wisdom becomes a garden.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9130404591560364
            ],
            "recall": [
                0.9768903255462646
            ],
            "f1": [
                0.9438868761062622
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9849439263343811
            ]
        },
        "COMET": {
            "scores": [
                0.9289984107017517
            ]
        }
    },
    {
        "ingles_original": "dream becomes a star",
        "portugues_traduzido": "sonho se torna uma estrela",
        "ingles_traduzido": "dream becomes a star",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 1.0,
            "precisions": [
                1.0,
                1.0,
                1.0,
                1.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 4,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0136955976486206
            ]
        },
        "COMET": {
            "scores": [
                0.9877099990844727
            ]
        }
    },
    {
        "ingles_original": "the chef writes the restaurant.",
        "portugues_traduzido": "o chef escreve o restaurante.",
        "ingles_traduzido": "The chef writes the restaurant.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.000000238418579
            ],
            "recall": [
                1.000000238418579
            ],
            "f1": [
                1.000000238418579
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0537066459655762
            ]
        },
        "COMET": {
            "scores": [
                0.9364845156669617
            ]
        }
    },
    {
        "ingles_original": "the driver visits the cafe.",
        "portugues_traduzido": "o motorista visita o café.",
        "ingles_traduzido": "The driver visits the café.",
        "ROUGE": {
            "rouge1": 0.8000000000000002,
            "rouge2": 0.75,
            "rougeL": 0.8000000000000002,
            "rougeLsum": 0.8000000000000002
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6666666666666666,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.0
            ],
            "recall": [
                1.0
            ],
            "f1": [
                1.0
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0576101541519165
            ]
        },
        "COMET": {
            "scores": [
                0.9417674541473389
            ]
        }
    },
    {
        "ingles_original": "word has a rock",
        "portugues_traduzido": "a palavra tem uma pedra",
        "ingles_traduzido": "the word has a stone",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.5714285714285715,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.87518310546875
            ],
            "recall": [
                0.9058449268341064
            ],
            "f1": [
                0.8902500867843628
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.0549517422914505
            ]
        },
        "COMET": {
            "scores": [
                0.8283203840255737
            ]
        }
    },
    {
        "ingles_original": "she writes the office.",
        "portugues_traduzido": "Ela escreve o escritório.",
        "ingles_traduzido": "She writes the office.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.668740304976422,
            "precisions": [
                0.8,
                0.75,
                0.6666666666666666,
                0.5
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 5,
            "reference_lenght": 5
        },
        "BERTSCORE": {
            "precision": [
                0.9999999403953552
            ],
            "recall": [
                0.9999999403953552
            ],
            "f1": [
                0.9999999403953552
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.040927767753601
            ]
        },
        "COMET": {
            "scores": [
                0.9256632924079895
            ]
        }
    },
    {
        "ingles_original": "the child drinks the market.",
        "portugues_traduzido": "A criança bebe o mercado.",
        "ingles_traduzido": "The child drinks the market.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.0
            ],
            "recall": [
                1.0
            ],
            "f1": [
                1.0
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.05592679977417
            ]
        },
        "COMET": {
            "scores": [
                0.9117934107780457
            ]
        }
    },
    {
        "ingles_original": "time is a mountain",
        "portugues_traduzido": "o tempo é uma montanha",
        "ingles_traduzido": "Time is a mountain.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9236505031585693
            ],
            "recall": [
                0.9790476560592651
            ],
            "f1": [
                0.950542688369751
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.939275324344635
            ]
        },
        "COMET": {
            "scores": [
                0.9523191452026367
            ]
        }
    },
    {
        "ingles_original": "wisdom has a garden",
        "portugues_traduzido": "A sabedoria tem um jardim.",
        "ingles_traduzido": "Wisdom has a garden.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9052119851112366
            ],
            "recall": [
                0.9658809900283813
            ],
            "f1": [
                0.9345629215240479
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9994896650314331
            ]
        },
        "COMET": {
            "scores": [
                0.9222677946090698
            ]
        }
    },
    {
        "ingles_original": "mind has a gold",
        "portugues_traduzido": "mente tem um ouro",
        "ingles_traduzido": "Mind has a gold.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9161999821662903
            ],
            "recall": [
                0.9626255631446838
            ],
            "f1": [
                0.9388391971588135
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0060608386993408
            ]
        },
        "COMET": {
            "scores": [
                0.9103217720985413
            ]
        }
    },
    {
        "ingles_original": "hope is a ice",
        "portugues_traduzido": "a esperança é um gelo",
        "ingles_traduzido": "Hope is a piece of ice.",
        "ROUGE": {
            "rouge1": 0.8,
            "rouge2": 0.5,
            "rougeL": 0.8,
            "rougeLsum": 0.8
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.16666666666666666,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8524298667907715
            ],
            "recall": [
                0.9082615971565247
            ],
            "f1": [
                0.8794605135917664
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.08137589693069458
            ]
        },
        "COMET": {
            "scores": [
                0.8842135667800903
            ]
        }
    },
    {
        "ingles_original": "heart is a tree",
        "portugues_traduzido": "coração é uma árvore",
        "ingles_traduzido": "The heart is a tree.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8516923785209656
            ],
            "recall": [
                0.9445767998695374
            ],
            "f1": [
                0.8957330584526062
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8557640314102173
            ]
        },
        "COMET": {
            "scores": [
                0.9255202412605286
            ]
        }
    },
    {
        "ingles_original": "spirit becomes a storm",
        "portugues_traduzido": "o espírito se torna uma tempestade",
        "ingles_traduzido": "\"The spirit becomes a storm.\"",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.345720784641941,
            "precisions": [
                0.5,
                0.42857142857142855,
                0.3333333333333333,
                0.2
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8516387343406677
            ],
            "recall": [
                0.9540252685546875
            ],
            "f1": [
                0.8999291658401489
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7931667566299438
            ]
        },
        "COMET": {
            "scores": [
                0.8971481323242188
            ]
        }
    },
    {
        "ingles_original": "dream is a star",
        "portugues_traduzido": "o sonho é uma estrela",
        "ingles_traduzido": "The dream is a star.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8622832894325256
            ],
            "recall": [
                0.9426438808441162
            ],
            "f1": [
                0.9006746411323547
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.3479841351509094
            ]
        },
        "COMET": {
            "scores": [
                0.9123738408088684
            ]
        }
    },
    {
        "ingles_original": "the artist works in the library.",
        "portugues_traduzido": "o artista trabalha na biblioteca.",
        "ingles_traduzido": "The artist works in the library.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.8091067115702212,
            "precisions": [
                0.8571428571428571,
                0.8333333333333334,
                0.8,
                0.75
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 7,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0512031316757202
            ]
        },
        "COMET": {
            "scores": [
                0.9326633214950562
            ]
        }
    },
    {
        "ingles_original": "smile is a wind",
        "portugues_traduzido": "Sorriso é um vento.",
        "ingles_traduzido": "A smile is a wind.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.880831241607666
            ],
            "recall": [
                0.9295331239700317
            ],
            "f1": [
                0.9045271277427673
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.6455758810043335
            ]
        },
        "COMET": {
            "scores": [
                0.9064367413520813
            ]
        }
    },
    {
        "ingles_original": "passion has a light",
        "portugues_traduzido": "a paixão tem uma luz",
        "ingles_traduzido": "\"Passion has a light.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.860945999622345
            ],
            "recall": [
                0.9604803323745728
            ],
            "f1": [
                0.9079935550689697
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8640905618667603
            ]
        },
        "COMET": {
            "scores": [
                0.8475838303565979
            ]
        }
    },
    {
        "ingles_original": "the child drinks the market.",
        "portugues_traduzido": "A criança bebe o mercado.",
        "ingles_traduzido": "The child drinks the market.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.0
            ],
            "recall": [
                1.0
            ],
            "f1": [
                1.0
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.05592679977417
            ]
        },
        "COMET": {
            "scores": [
                0.9117934107780457
            ]
        }
    },
    {
        "ingles_original": "smile is a wind",
        "portugues_traduzido": "sorriso é um vento",
        "ingles_traduzido": "\"smooth is a wind\"",
        "ROUGE": {
            "rouge1": 0.75,
            "rouge2": 0.6666666666666666,
            "rougeL": 0.75,
            "rougeLsum": 0.75
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.5,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8303154706954956
            ],
            "recall": [
                0.9110177159309387
            ],
            "f1": [
                0.8687965273857117
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                -0.4262982904911041
            ]
        },
        "COMET": {
            "scores": [
                0.6935119032859802
            ]
        }
    },
    {
        "ingles_original": "hope has a ice",
        "portugues_traduzido": "a esperança tem um gelo",
        "ingles_traduzido": "Hope has a freeze.",
        "ROUGE": {
            "rouge1": 0.75,
            "rouge2": 0.6666666666666666,
            "rougeL": 0.75,
            "rougeLsum": 0.75
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.4,
                0.25,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8694033026695251
            ],
            "recall": [
                0.9173908233642578
            ],
            "f1": [
                0.8927526473999023
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                -0.1248779222369194
            ]
        },
        "COMET": {
            "scores": [
                0.7454856038093567
            ]
        }
    },
    {
        "ingles_original": "hope is a ice",
        "portugues_traduzido": "A esperança é um gelo.",
        "ingles_traduzido": "Hope is ice.",
        "ROUGE": {
            "rouge1": 0.8571428571428571,
            "rouge2": 0.4,
            "rougeL": 0.8571428571428571,
            "rougeLsum": 0.8571428571428571
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.5,
                0.0,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 4,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8622839450836182
            ],
            "recall": [
                0.8999738693237305
            ],
            "f1": [
                0.8807259202003479
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                -0.05238097906112671
            ]
        },
        "COMET": {
            "scores": [
                0.8920938372612
            ]
        }
    },
    {
        "ingles_original": "the driver visits the cafe.",
        "portugues_traduzido": "o motorista visita o café.",
        "ingles_traduzido": "The driver visits the café.",
        "ROUGE": {
            "rouge1": 0.8000000000000002,
            "rouge2": 0.75,
            "rougeL": 0.8000000000000002,
            "rougeLsum": 0.8000000000000002
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6666666666666666,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.0
            ],
            "recall": [
                1.0
            ],
            "f1": [
                1.0
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0576101541519165
            ]
        },
        "COMET": {
            "scores": [
                0.9417674541473389
            ]
        }
    },
    {
        "ingles_original": "time has a mountain",
        "portugues_traduzido": "o tempo tem uma montanha",
        "ingles_traduzido": "\"time has a mountain\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8723601698875427
            ],
            "recall": [
                0.9748002290725708
            ],
            "f1": [
                0.9207396507263184
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.814486026763916
            ]
        },
        "COMET": {
            "scores": [
                0.8496859073638916
            ]
        }
    },
    {
        "ingles_original": "eye has a flame",
        "portugues_traduzido": "o olho tem uma chama",
        "ingles_traduzido": "the eye has a flame",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.668740304976422,
            "precisions": [
                0.8,
                0.75,
                0.6666666666666666,
                0.5
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9360626935958862
            ],
            "recall": [
                0.9753048419952393
            ],
            "f1": [
                0.9552809000015259
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9751796722412109
            ]
        },
        "COMET": {
            "scores": [
                0.9536128640174866
            ]
        }
    },
    {
        "ingles_original": "silence is a thunder",
        "portugues_traduzido": "o silêncio é um trovão",
        "ingles_traduzido": "Silence is a thunder.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.941550612449646
            ],
            "recall": [
                0.9675469994544983
            ],
            "f1": [
                0.9543717503547668
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0127891302108765
            ]
        },
        "COMET": {
            "scores": [
                0.8793434500694275
            ]
        }
    },
    {
        "ingles_original": "courage becomes a sun",
        "portugues_traduzido": "A coragem se torna um sol.",
        "ingles_traduzido": "\"Courage becomes a sun.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8657581210136414
            ],
            "recall": [
                0.9646015167236328
            ],
            "f1": [
                0.9125109314918518
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8006837964057922
            ]
        },
        "COMET": {
            "scores": [
                0.8619086146354675
            ]
        }
    },
    {
        "ingles_original": "passion becomes a light",
        "portugues_traduzido": "a paixão se torna uma luz",
        "ingles_traduzido": "\"Passion becomes a light.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8640274405479431
            ],
            "recall": [
                0.9667182564735413
            ],
            "f1": [
                0.9124928116798401
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8607560396194458
            ]
        },
        "COMET": {
            "scores": [
                0.9112815260887146
            ]
        }
    },
    {
        "ingles_original": "word has a rock",
        "portugues_traduzido": "A palavra tem uma pedra.",
        "ingles_traduzido": "The word has a stone.",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.5714285714285715,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.5,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8477987051010132
            ],
            "recall": [
                0.8775043487548828
            ],
            "f1": [
                0.8623957633972168
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.03087671846151352
            ]
        },
        "COMET": {
            "scores": [
                0.7778559327125549
            ]
        }
    },
    {
        "ingles_original": "word is a rock",
        "portugues_traduzido": "A palavra é uma rocha.",
        "ingles_traduzido": "The word is a rock.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8598257303237915
            ],
            "recall": [
                0.8922830820083618
            ],
            "f1": [
                0.8757538199424744
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8364371657371521
            ]
        },
        "COMET": {
            "scores": [
                0.8961144089698792
            ]
        }
    },
    {
        "ingles_original": "courage has a sun",
        "portugues_traduzido": "A coragem tem um sol.",
        "ingles_traduzido": "\"Courage has a sun.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8626562356948853
            ],
            "recall": [
                0.9582747220993042
            ],
            "f1": [
                0.9079549312591553
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8209140300750732
            ]
        },
        "COMET": {
            "scores": [
                0.8216587901115417
            ]
        }
    },
    {
        "ingles_original": "the chef writes the restaurant.",
        "portugues_traduzido": "o chef escreve o restaurante.",
        "ingles_traduzido": "The chef writes the restaurant.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.7598356856515925,
            "precisions": [
                0.8333333333333334,
                0.8,
                0.75,
                0.6666666666666666
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.000000238418579
            ],
            "recall": [
                1.000000238418579
            ],
            "f1": [
                1.000000238418579
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0537066459655762
            ]
        },
        "COMET": {
            "scores": [
                0.9364845156669617
            ]
        }
    },
    {
        "ingles_original": "life is a river",
        "portugues_traduzido": "A vida é um rio.",
        "ingles_traduzido": "Life is a river.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8987619876861572
            ],
            "recall": [
                0.9779398441314697
            ],
            "f1": [
                0.9366806745529175
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9836459755897522
            ]
        },
        "COMET": {
            "scores": [
                0.9609339833259583
            ]
        }
    },
    {
        "ingles_original": "life is a river",
        "portugues_traduzido": "A vida é um rio.",
        "ingles_traduzido": "Life is a river.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8987619876861572
            ],
            "recall": [
                0.9779398441314697
            ],
            "f1": [
                0.9366806745529175
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9836459755897522
            ]
        },
        "COMET": {
            "scores": [
                0.9609339833259583
            ]
        }
    },
    {
        "ingles_original": "silence has a thunder",
        "portugues_traduzido": "o silêncio tem um trovão",
        "ingles_traduzido": "Silence has a thunder.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9439051747322083
            ],
            "recall": [
                0.9670805931091309
            ],
            "f1": [
                0.9553523659706116
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0000044107437134
            ]
        },
        "COMET": {
            "scores": [
                0.8715032935142517
            ]
        }
    },
    {
        "ingles_original": "life has a river",
        "portugues_traduzido": "A vida tem um rio.",
        "ingles_traduzido": "Life has a river.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.911210298538208
            ],
            "recall": [
                0.9720832109451294
            ],
            "f1": [
                0.9406629800796509
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0252938270568848
            ]
        },
        "COMET": {
            "scores": [
                0.9469362497329712
            ]
        }
    },
    {
        "ingles_original": "memory is a ocean",
        "portugues_traduzido": "A memória é um oceano.",
        "ingles_traduzido": "Memory is an ocean.",
        "ROUGE": {
            "rouge1": 0.75,
            "rouge2": 0.3333333333333333,
            "rougeL": 0.75,
            "rougeLsum": 0.75
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.4,
                0.0,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8953099250793457
            ],
            "recall": [
                0.9554542303085327
            ],
            "f1": [
                0.9244048595428467
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8938137888908386
            ]
        },
        "COMET": {
            "scores": [
                0.9381778836250305
            ]
        }
    },
    {
        "ingles_original": "voice becomes a iron",
        "portugues_traduzido": "\"a voz se torna um ferro\"",
        "ingles_traduzido": "\"the voice becomes an iron\"",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.28571428571428575,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.16666666666666666,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.85211580991745
            ],
            "recall": [
                0.9309350252151489
            ],
            "f1": [
                0.8897833824157715
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.5163276195526123
            ]
        },
        "COMET": {
            "scores": [
                0.9002875089645386
            ]
        }
    },
    {
        "ingles_original": "smile has a wind",
        "portugues_traduzido": "o sorriso tem um vento",
        "ingles_traduzido": "\"The smile has a wind.\"",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.345720784641941,
            "precisions": [
                0.5,
                0.42857142857142855,
                0.3333333333333333,
                0.2
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8371331691741943
            ],
            "recall": [
                0.9254881143569946
            ],
            "f1": [
                0.8790961503982544
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.6899816393852234
            ]
        },
        "COMET": {
            "scores": [
                0.881645143032074
            ]
        }
    },
    {
        "ingles_original": "heart becomes a tree",
        "portugues_traduzido": "coração se torna uma árvore",
        "ingles_traduzido": "\"Heart becomes a tree.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8608188033103943
            ],
            "recall": [
                0.9686713814735413
            ],
            "f1": [
                0.9115660190582275
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8788118362426758
            ]
        },
        "COMET": {
            "scores": [
                0.8349569439888
            ]
        }
    },
    {
        "ingles_original": "word is a rock",
        "portugues_traduzido": "A palavra é uma rocha.",
        "ingles_traduzido": "The word is a rock.",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8598257303237915
            ],
            "recall": [
                0.8922830820083618
            ],
            "f1": [
                0.8757538199424744
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8364371657371521
            ]
        },
        "COMET": {
            "scores": [
                0.8961144089698792
            ]
        }
    },
    {
        "ingles_original": "memory has a ocean",
        "portugues_traduzido": "a memória tem um oceano",
        "ingles_traduzido": "\"The memory has an ocean.\"",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.28571428571428575,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.375,
                0.14285714285714285,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8434924483299255
            ],
            "recall": [
                0.9323244690895081
            ],
            "f1": [
                0.8856866359710693
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.5097768306732178
            ]
        },
        "COMET": {
            "scores": [
                0.884214460849762
            ]
        }
    },
    {
        "ingles_original": "the driver visits the cafe.",
        "portugues_traduzido": "O motorista visita o café.",
        "ingles_traduzido": "The driver visits the café.",
        "ROUGE": {
            "rouge1": 0.8000000000000002,
            "rouge2": 0.75,
            "rougeL": 0.8000000000000002,
            "rougeLsum": 0.8000000000000002
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6666666666666666,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                1.0
            ],
            "recall": [
                1.0
            ],
            "f1": [
                1.0
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0576101541519165
            ]
        },
        "COMET": {
            "scores": [
                0.9420997500419617
            ]
        }
    },
    {
        "ingles_original": "courage becomes a sun",
        "portugues_traduzido": "A coragem se torna um sol.",
        "ingles_traduzido": "\"Courage becomes a sun.\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.42857142857142855,
                0.3333333333333333,
                0.2,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.75,
            "translation_lenght": 7,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8657581210136414
            ],
            "recall": [
                0.9646015167236328
            ],
            "f1": [
                0.9125109314918518
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8006837964057922
            ]
        },
        "COMET": {
            "scores": [
                0.8619086146354675
            ]
        }
    },
    {
        "ingles_original": "silence has a thunder",
        "portugues_traduzido": "o silêncio tem um trovão",
        "ingles_traduzido": "Silence has a thunder.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9439051747322083
            ],
            "recall": [
                0.9670805931091309
            ],
            "f1": [
                0.9553523659706116
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0000044107437134
            ]
        },
        "COMET": {
            "scores": [
                0.8715032935142517
            ]
        }
    },
    {
        "ingles_original": "spirit becomes a storm",
        "portugues_traduzido": "o espírito se torna uma tempestade",
        "ingles_traduzido": "the spirit becomes a storm",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.668740304976422,
            "precisions": [
                0.8,
                0.75,
                0.6666666666666666,
                0.5
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9455221891403198
            ],
            "recall": [
                0.9809587597846985
            ],
            "f1": [
                0.9629145264625549
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.9372416138648987
            ]
        },
        "COMET": {
            "scores": [
                0.9419514536857605
            ]
        }
    },
    {
        "ingles_original": "life becomes a river",
        "portugues_traduzido": "A vida se torna um rio.",
        "ingles_traduzido": "Life becomes a river.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6,
                0.5,
                0.3333333333333333,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.25,
            "translation_lenght": 5,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.9092971086502075
            ],
            "recall": [
                0.9771600365638733
            ],
            "f1": [
                0.9420079588890076
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.003830075263977
            ]
        },
        "COMET": {
            "scores": [
                0.9733403325080872
            ]
        }
    },
    {
        "ingles_original": "time becomes a mountain",
        "portugues_traduzido": "o tempo se torna uma montanha",
        "ingles_traduzido": "\"time becomes a mountain\"",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.5081327481546147,
            "precisions": [
                0.6666666666666666,
                0.6,
                0.5,
                0.3333333333333333
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.5,
            "translation_lenght": 6,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8587406873703003
            ],
            "recall": [
                0.9695554375648499
            ],
            "f1": [
                0.9107897281646729
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.8432538509368896
            ]
        },
        "COMET": {
            "scores": [
                0.9474061727523804
            ]
        }
    },
    {
        "ingles_original": "memory has a ocean",
        "portugues_traduzido": "A memória tem um oceano.",
        "ingles_traduzido": "\"The memory has an ocean.\"",
        "ROUGE": {
            "rouge1": 0.6666666666666665,
            "rouge2": 0.28571428571428575,
            "rougeL": 0.6666666666666665,
            "rougeLsum": 0.6666666666666665
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.375,
                0.14285714285714285,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8434924483299255
            ],
            "recall": [
                0.9323244690895081
            ],
            "f1": [
                0.8856866359710693
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.5097768306732178
            ]
        },
        "COMET": {
            "scores": [
                0.8783395290374756
            ]
        }
    },
    {
        "ingles_original": "spirit is a storm",
        "portugues_traduzido": "O espírito é uma tempestade.",
        "ingles_traduzido": "\"The spirit is a storm.\"",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.345720784641941,
            "precisions": [
                0.5,
                0.42857142857142855,
                0.3333333333333333,
                0.2
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8424568176269531
            ],
            "recall": [
                0.948248565196991
            ],
            "f1": [
                0.8922276496887207
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7469356060028076
            ]
        },
        "COMET": {
            "scores": [
                0.874889612197876
            ]
        }
    },
    {
        "ingles_original": "silence has a thunder",
        "portugues_traduzido": "o silêncio tem um trovão",
        "ingles_traduzido": "\"The silence has a thunder.\"",
        "ROUGE": {
            "rouge1": 0.888888888888889,
            "rouge2": 0.8571428571428571,
            "rougeL": 0.888888888888889,
            "rougeLsum": 0.888888888888889
        },
        "BLEU": {
            "bleu": 0.345720784641941,
            "precisions": [
                0.5,
                0.42857142857142855,
                0.3333333333333333,
                0.2
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 2.0,
            "translation_lenght": 8,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.8536924123764038
            ],
            "recall": [
                0.9484155178070068
            ],
            "f1": [
                0.8985645174980164
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.6480779647827148
            ]
        },
        "COMET": {
            "scores": [
                0.9085579514503479
            ]
        }
    },
    {
        "ingles_original": "mind becomes a gold",
        "portugues_traduzido": "a mente se torna ouro",
        "ingles_traduzido": "the mind becomes gold",
        "ROUGE": {
            "rouge1": 0.75,
            "rouge2": 0.3333333333333333,
            "rougeL": 0.75,
            "rougeLsum": 0.75
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.75,
                0.3333333333333333,
                0.0,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 4,
            "reference_lenght": 4
        },
        "BERTSCORE": {
            "precision": [
                0.920985221862793
            ],
            "recall": [
                0.9224328994750977
            ],
            "f1": [
                0.9217085242271423
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                0.7021422386169434
            ]
        },
        "COMET": {
            "scores": [
                0.9479367136955261
            ]
        }
    },
    {
        "ingles_original": "the doctor walks to the school.",
        "portugues_traduzido": "O médico caminha para a escola.",
        "ingles_traduzido": "The doctor walks to the school.",
        "ROUGE": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0
        },
        "BLEU": {
            "bleu": 0.8091067115702212,
            "precisions": [
                0.8571428571428571,
                0.8333333333333334,
                0.8,
                0.75
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 7,
            "reference_lenght": 7
        },
        "BERTSCORE": {
            "precision": [
                1.0000001192092896
            ],
            "recall": [
                1.0000001192092896
            ],
            "f1": [
                1.0000001192092896
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                1.0738896131515503
            ]
        },
        "COMET": {
            "scores": [
                0.9442793726921082
            ]
        }
    },
    {
        "ingles_original": "my friend drinks the store.",
        "portugues_traduzido": "Meu amigo toma a loja.",
        "ingles_traduzido": "My friend takes the store.",
        "ROUGE": {
            "rouge1": 0.8000000000000002,
            "rouge2": 0.5,
            "rougeL": 0.8000000000000002,
            "rougeLsum": 0.8000000000000002
        },
        "BLEU": {
            "bleu": 0.0,
            "precisions": [
                0.6666666666666666,
                0.4,
                0.25,
                0.0
            ],
            "brevity_penalty": 1.0,
            "length_ratio": 1.0,
            "translation_lenght": 6,
            "reference_lenght": 6
        },
        "BERTSCORE": {
            "precision": [
                0.9483228325843811
            ],
            "recall": [
                0.9483228325843811
            ],
            "f1": [
                0.9483228325843811
            ],
            "hashcode": "distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.50.3)"
        },
        "BLEURT": {
            "scores": [
                -0.48557963967323303
            ]
        },
        "COMET": {
            "scores": [
                0.7212011814117432
            ]
        }
    }
]

for objeto in dados[114:]:
    prediction = objeto["ingles_traduzido"]
    reference = objeto["ingles_original"]
    source = objeto["portugues_traduzido"]

    result_rouge = rouge.compute(predictions=[prediction], references=[reference])
    result_bleu = bleu.compute(predictions=[prediction], references=[reference])
    result_bertscore = bertscore.compute(predictions=[prediction], references=[reference], model_type="distilbert-base-uncased")
    result_bleurt = bleurt.compute(predictions=[prediction], references=[reference])
    result_comet = comet.compute(predictions=[prediction], references=[reference], sources=[source])

    result = {
        "ingles_original": objeto["ingles_original"],
        "portugues_traduzido": objeto["portugues_traduzido"],
        "ingles_traduzido": objeto["ingles_traduzido"],
        "ROUGE": {
            "rouge1" : result_rouge["rouge1"],
            "rouge2" : result_rouge["rouge2"],
            "rougeL" : result_rouge["rougeL"],
            "rougeLsum" : result_rouge["rougeLsum"],
        },
        "BLEU": {
            "bleu" : result_bleu["bleu"],
            "precisions" : result_bleu["precisions"],
            "brevity_penalty" : result_bleu["brevity_penalty"],
            "length_ratio" : result_bleu["length_ratio"],
            "translation_lenght" : result_bleu["translation_length"],
            "reference_lenght" : result_bleu["reference_length"]
        },
        "BERTSCORE": {
            "precision" : result_bertscore["precision"],
            "recall" : result_bertscore["recall"],
            "f1" : result_bertscore["f1"],
            "hashcode" : result_bertscore["hashcode"]
        },
        "BLEURT": {
            "scores" : result_bleurt["scores"],

        },
        "COMET": {
            "scores" : result_comet["scores"],
        }
    }

    vetor.append(result)

    with open('gpt/frases_traduzidas_com_metricas.json', 'w', encoding='utf-8') as file:
        json.dump(vetor, file, ensure_ascii=False, indent=4)

