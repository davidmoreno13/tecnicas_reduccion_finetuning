import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from capas_gpt import TransformerBlock, LayerNorm
import copy
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}
    
with open("config_gpt.json", "r") as f:
    cfg = json.load(f)

model_path = "modelo_gpt_custom.pth"

model_base = GPTModel(cfg)
model_base.load_state_dict(torch.load(model_path, map_location="cpu"))
model_base.eval()

import random

def generate_text(model,tokenizer,prompt,seed=42,max_new_tokens=50,temperature=0.9,top_k=50,top_p=0.95,repetition_penalty=1.1):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        input_ids_cropped = generated_ids[:, -cfg["context_length"]:]

        with torch.no_grad():
            outputs = model(input_ids=input_ids_cropped)
            logits = outputs["logits"][:, -1, :]

        for token_id in set(generated_ids[0].tolist()):
            logits[0, token_id] /= repetition_penalty

        logits = logits / temperature

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            threshold = values[:, -1].unsqueeze(-1)
            logits[logits < threshold] = -float("Inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return output_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_base.to(device);

import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and any(x in name.lower() for x in ["q", "k", "v", "proj", "fc"]):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)

model_lora = copy.deepcopy(model_base)

for param in model_lora.parameters():
    param.requires_grad = False

replace_linear_with_lora(model_lora, rank=16, alpha=16)

model_lora.load_state_dict(torch.load("modelo_lora.pth", map_location=device))

logging.basicConfig(level=logging.INFO)

user_sessions = {}

def generar_opciones(model, tokenizer, contexto, paso, tokens_por_paso=40):
    opcion1 = generate_text(model, tokenizer, contexto, max_new_tokens=tokens_por_paso,
                            temperature=0.9, top_k=40, seed=paso * 2)
    opcion2 = generate_text(model, tokenizer, contexto, max_new_tokens=tokens_por_paso,
                            temperature=1.1, top_k=40, seed=paso * 2 + 1)
    texto1 = opcion1[len(contexto):].strip()
    texto2 = opcion2[len(contexto):].strip()
    return texto1, texto2, opcion1.strip(), opcion2.strip()

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensaje = (
    "📚 <b>Bienvenido a GPTales, el generador interactivo de historias con IA</b>\n\n"
    "Este bot no usa una IA cualquiera. GPTales funciona gracias a un modelo <b>GPT creado desde cero</b> en PyTorch, "
    "inspirado en la arquitectura original de OpenAI. Todo el código del modelo ha sido implementado manualmente, "
    "incluyendo los bloques Transformer, capas de atención y embeddings.\n\n"
    "Además, incluye una versión especializada entrenada con <b>LoRA</b> (Low-Rank Adaptation), un método de fine-tuning "
    "más eficiente. Este fine-tuning también ha sido aplicado manualmente, sin usar librerías externas como PEFT o HuggingFace. "
    "El modelo fue afinado con un <b>dataset de historias de terror</b>, lo que da lugar a una versión más oscura y narrativa del GPT original.\n\n"
    "🎮 <b>¿Cómo funciona?</b>\n"
    "1. Inicias la historia (modo normal o terror).\n"
    "2. El bot te genera un fragmento y dos posibles continuaciones.\n"
    "3. Tú eliges cómo continuar.\n"
    "4. La historia avanza según tus decisiones.\n\n"
    "También puedes escribir <b>IA</b> como inicio para que el bot genere un comienzo automáticamente.\n\n"
    "🧭 <b>Comandos disponibles:</b>\n"
    "/start - Iniciar una nueva historia\n"
    "/normal - Modo historia general\n"
    "/terror - Modo historia de terror\n"
    "/fin - Finalizar la historia actual\n"
    "/help - Ver esta explicación y los comandos"
    )

    await update.message.reply_text(mensaje, parse_mode="HTML")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_sessions[chat_id] = {"step": 0, "text": "", "mode": ""}
    await update.message.reply_text("👋 ¡Hola! ¿Quieres generar una historia normal o de terror? Usa el comando /normal o /terror para elegir el modo deseado.")

async def normal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_sessions[chat_id]["mode"] = "normal"
    await pedir_prompt(update)

async def terror(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_sessions[chat_id]["mode"] = "terror"
    await pedir_prompt(update)

async def fin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = user_sessions.get(chat_id)
    if session and session.get("text"):
        historia_completa = session["text"]
        await update.message.reply_text("📚 *Historia completa:*", parse_mode="Markdown")
        bloques = [historia_completa[i:i+4000] for i in range(0, len(historia_completa), 4000)]
        for bloque in bloques:
            await update.message.reply_text(bloque)
        await update.message.reply_text("✅ Has finalizado tu historia. Usa /start para comenzar una nueva.")
        del user_sessions[chat_id]
    else:
        await update.message.reply_text("❌ No tienes ninguna historia activa. Usa /start para comenzar.")

async def pedir_prompt(update: Update):
    await update.message.reply_text("🪄 Escribe el inicio de tu historia (o escribe 'IA' para que la empiece automáticamente la IA):")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    mensaje = update.message.text.strip()
    session = user_sessions.get(chat_id)
    if not session or session.get("mode") == "":
        await update.message.reply_text("Usa /start primero 😊")
        return
    if mensaje.lower() in ["fin", "/fin"]:
        await fin(update, context)
        return
    if session["text"] == "":
        if mensaje.lower() == "ia":
            prompts = [
                "It was a quiet night until the phone rang unexpectedly.",
                "Deep in the forest, something ancient had awakened.",
                "She never expected the letter to arrive after all these years.",
                "The sky turned red as the city fell silent.",
                "I was walking home when I saw the shadow move.",
                "The mirror in the attic began to whisper again.",
                "No one believed him when he said he saw a ghost at school.",
                "Every night, the same dream. Every night, a little closer.",
                "They thought it was just a power outage... until the screams began.",
                "He opened the door and there it was — not human, not anymore."
            ]
            prompt = random.choice(prompts)
        else:
            prompt = mensaje
        session["text"] = prompt
        session["step"] = 0
        await continuar_historia(update, context, session)
    else:
        if mensaje.startswith("2"):
            session["text"] = session["opcion2"]
        else:
            session["text"] = session["opcion1"]
        session["step"] += 1
        await continuar_historia(update, context, session)

async def continuar_historia(update: Update, context: ContextTypes.DEFAULT_TYPE, session):
    model = model_lora if session["mode"] == "terror" else model_base

    historia_formateada = session["text"]
    await update.message.reply_text(f"📖 *Historia hasta ahora:* {historia_formateada}...", parse_mode="Markdown")

    await update.message.reply_text("✅ *Escribe 1 o 2 para elegir cómo continuar la historia.*\n❌ También puedes escribir *Fin* o usar el comando /fin para finalizar la historia.", parse_mode="Markdown")

    texto1, texto2, full1, full2 = generar_opciones(model, tokenizer, session["text"], paso=session["step"])
    session["opcion1"] = full1
    session["opcion2"] = full2

    await update.message.reply_text("1️⃣ *Opción 1:*", parse_mode="Markdown")
    await update.message.reply_text(f"...{texto1}")

    await update.message.reply_text("2️⃣ *Opción 2:*", parse_mode="Markdown")
    await update.message.reply_text(f"...{texto2}")

# Token del bot
TOKEN = "7874960165:AAFFlGBsJJfwBLLuX4Nrf31x-w7tYCJshIY"

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("normal", normal))
app.add_handler(CommandHandler("terror", terror))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CommandHandler("fin", fin))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

if __name__ == "__main__":
    print("Bot corriendo...")
    app.run_polling()
