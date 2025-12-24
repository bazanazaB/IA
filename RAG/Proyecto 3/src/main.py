import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_script(relative_path):
    script_path = BASE_DIR / relative_path
    print(f"\n===========================")
    print(f" Ejecutando: {script_path.name}")
    print(f"===========================\n")
    subprocess.run(["python", str(script_path)], check=True)


def main():
    print("\n====================================")
    print("   PIPELINE COMPLETO RAG INICIADO   ")
    print("====================================\n")

  # run_script("preprocessing/limpieza.py")
    run_script("embeddings/embeddings.py")
    run_script("retrieval/vectorstore.py")
    run_script("analysis/mapa_conceptual.py")
    run_script("retrieval/rag.py")

    print("\n====================================")
    print("    PIPELINE COMPLETO FINALIZADO    ")
    print("====================================\n")


if __name__ == "__main__":
    main()
