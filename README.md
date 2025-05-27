# 🧠 GPTales: Generador de Historias Interactivo con GPT-2 y LoRA

Este proyecto corresponde al Trabajo de Fin de Grado del Doble Grado en Ingeniería Informática y Matemáticas (Universidad de Sevilla), y tiene como objetivo principal la **implementación desde cero de un modelo GPT-2 en PyTorch** y su **adaptación mediante la técnica Low-Rank Adaptation (LoRA)** para modificar su estilo narrativo.

A partir de esta base, se ha construido un generador interactivo de historias con diferentes tonos (general o terror), accesible a través de un **bot de Telegram**.

---

## 📁 Estructura del repositorio

| Archivo | Descripción |
|--------|-------------|
| `0_capas_gpt.py` | Implementación de las capas básicas del modelo GPT: atención multi-cabeza, feed-forward, normalización por capas y bloques Transformer. |
| `1_descarga_model_base.ipynb` | Construcción de la arquitectura base, carga del tokenizer oficial de GPT-2, importación de pesos preentrenados y prueba inicial de generación de texto. |
| `2_impl_model_base.ipynb` | Desarrollo de una función generadora de historias autoregresiva, incluyendo control sobre temperatura, top-k y otros hiperparámetros. |
| `3_impl_lora.ipynb` | Implementación manual de LoRA para insertar capas de bajo rango en las matrices clave del modelo, permitiendo un fine-tuning eficiente. |
| `4_entrenamiento_lora.ipynb` | Preprocesamiento de un dataset de historias de terror, tokenización, entrenamiento del modelo LoRA y evaluación básica del resultado. |
| `5_bot_historia_telegram.py` | Bot de Telegram completo que permite al usuario generar historias interactivas, eligiendo entre estilo general o de terror. |
| `6_metricas.ipynb` | Cálculo de métricas para comparar el modelo base y el ajustado con LoRA: perplexity, densidad léxica, KL Divergence, Cosine Similarity, etc. |
