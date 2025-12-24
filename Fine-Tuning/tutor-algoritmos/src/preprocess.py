import json
import random
from pathlib import Path

# ===============================
# RUTAS
# ===============================

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")

RAW_FILE = RAW_DIR / "ejemplos.md"

TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"
TEST_FILE = PROCESSED_DIR / "test.jsonl"

# ===============================
# PROPORCIONES
# ===============================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def parse_markdown(md_text: str):
    """
    Extrae pares instruction / output desde el markdown
    """
    examples = []
    blocks = md_text.split("---")

    for block in blocks:
        if "### Instrucción" in block and "### Respuesta" in block:
            try:
                instruction = block.split("### Instrucción")[1].split("### Respuesta")[0].strip()
                output = block.split("### Respuesta")[1].strip()

                if instruction and output:
                    examples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    })
            except IndexError:
                continue

    return examples


def save_jsonl(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def main():
    print(">>> Leyendo ejemplos.md...")
    if not RAW_FILE.exists():
        print(f"❌ No se encontró el archivo: {RAW_FILE}")
        return

    md_text = RAW_FILE.read_text(encoding="utf-8")

    examples = parse_markdown(md_text)
    print(f">>> Ejemplos encontrados: {len(examples)}")

    if len(examples) < 50:
        print("⚠️ Se recomienda al menos 100 ejemplos para mejores resultados.")

    random.shuffle(examples)

    total = len(examples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = examples[:train_end]
    val_data = examples[train_end:val_end]
    test_data = examples[val_end:]

    print(f">>> Train: {len(train_data)}")
    print(f">>> Val: {len(val_data)}")
    print(f">>> Test: {len(test_data)}")

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(val_data, VAL_FILE)
    save_jsonl(test_data, TEST_FILE)

    print("✅ Preprocesamiento completado")
    print("Archivos guardados en data/processed")


if __name__ == "__main__":
    main()
