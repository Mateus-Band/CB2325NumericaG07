#Arquivo para dia da aplicação

#ajuste de caminho para importação do projeto.
import sys
import os
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)