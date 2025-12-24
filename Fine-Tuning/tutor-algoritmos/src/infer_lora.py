import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_PATH = "../models/tutor-lora"

# ===============================
# Carga del modelo
# ===============================

def load_model():
    print(">> Cargando modelo base...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None,          # 🔥 DESACTIVA DISPATCH
        low_cpu_mem_usage=False   # 🔥 evita offload interno
    )

    model.to(device)

    print(">> Cargando LoRA...")
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
        device_map=None           # 🔥 MUY IMPORTANTE
    )

    model.eval()
    return tokenizer, model


# ===============================
# Inferencia pedagógica
# ===============================

def infer(prompt, tokenizer, model):
    template = f"""
### Rol:
Eres un tutor educativo experto.
Explicas de forma clara, precisa y pedagógica.
Respondes paso a paso y con ejemplos simples.

### Instrucción:
{prompt}

### Respuesta:
"""

    inputs = tokenizer(template, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.2,
            top_p=0.85,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Respuesta:" in result:
        result = result.split("### Respuesta:")[1].strip()

    return result


# ===============================
# Consola
# ===============================

def main():
    tokenizer, model = load_model()

    print("\n=== Tutor IA (modo pedagógico) ===\n")

    while True:
        user_input = input("Tú: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("\nHasta luego 👋")
            break

        respuesta = infer(user_input, tokenizer, model)
        print("\nTutor IA:\n", respuesta, "\n")


if __name__ == "__main__":
    main()
