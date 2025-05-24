# 🧠 LangGraph Agent Workshop

Este repositorio contiene ejemplos prácticos de agentes construidos con [LangGraph](https://github.com/langchain-ai/langgraph) en **Python**. LangGraph permite construir flujos de trabajo controlados, reactivos y colaborativos sobre LLMs de forma modular y clara.
---

## 📚 Recursos

* [Documentación oficial de LangGraph](https://docs.langgraph.dev/)
* [LangChain (complemento esencial)](https://docs.langchain.com/docs/)
* [Serper para búsquedas web](https://serper.dev/)

---

## 📦 Requisitos

* **Python 3.12+**
* Un proveedor de LLMs como **Watsonx** (puede ser otro backend compatible con LangChain)
* Una cuenta gratuita en [Serper](https://serper.dev/) para búsquedas en Google.

---

## 🛠️ Para empezar

1. **Clona este repositorio**

   ```bash
   git clone https://github.com/taller-genai-ucmc-mayo-2025/agentes-langgraph-taller
   cd langgraph-agent-workshop
   ```

2. **Instala las dependencias**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configura tu archivo `.env`**
   Crea un archivo llamado `.env` en la raíz del proyecto con las siguientes variables de entorno:

   ```env
   # IBM WatsonX
   WATSONX_APIKEY='tu_apikey'
   WATSONX_PROJECT_ID='tu_project_id'
   WATSONX_URL='https://us-south.ml.cloud.ibm.com'

   # Serper para búsqueda web
   SERPER_API_KEY='tu_clave_serper'
   USER_AGENT='usuarioResearch'
   ```

   > 📝 Puedes obtener tu clave gratuita de Serper en: [https://serper.dev/](https://serper.dev/)

4. **Ejecuta un agente de ejemplo**

   ```bash
   python src/simple_agent.py
   ```

---

## 🤖 ¿Qué contiene este repositorio?

* `src/`: agentes de ejemplo con diferentes arquitecturas y tareas
* `src/utils/`: herramientas auxiliares para interactuar en consola
* `.env.example`: plantilla para configurar variables de entorno
