#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/Users/luisa.stellet/Library/CloudStorage/GoogleDrive-luisastellet@id.uff.br/Meu Drive/Iniciacao Cientifica/metaphorPTback')

# Executar o script de análise
result = subprocess.run([sys.executable, 'scripts/analise_quartis_prompts_modelos.py'], 
                       capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print(f"Exit code: {result.returncode}")
