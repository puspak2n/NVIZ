# NarraViz 📊✨

**AI-Powered Data Storytelling & Visualization Engine**

NarraViz empowers users to transform raw datasets into clear, concise, and beautifully designed charts, summaries, and business insights — instantly. Built with Streamlit, Plotly, and OpenAI, it's your storytelling assistant for analytics.

---

## 🚀 Features

- ✅ AI Prompt Parser using Agentic Workflow
- 📊 Smart Chart Generation (Bar, Line, Pie, Scatter, etc.)
- 🎨 Professionally Styled Themes (Dark Mode, Business, Corporate)
- 📈 Outlier Detection, Trendlines, Temporal Analysis
- 💡 Insight Generator (OpenAI + Rule-based fallback)
- 🧠 Prompt memory & chart history
- 🔄 Toggle AI code generation on/off
- 🧪 Fallback charts when AI fails
- 💾 Snowflake & Supabase compatible
- 🛡️ Secure secrets management via `.streamlit/secrets.toml`

---

## 🧰 Tech Stack

| Tech | Purpose |
|------|---------|
| **Streamlit** | Web UI and layout |
| **Plotly** | Interactive charting |
| **OpenAI** | AI-powered insight generation |
| **Supabase** | Data store (current) |
| **Snowflake** | Enterprise-ready DB (optional) |
| **Pandas** | Data manipulation |
| **Python 3.10+** | Runtime |

---

## 🔧 Installation

```bash
git clone https://github.com/puspak2n/Narra.git
cd NarraViz
pip install -r requirements.txt
streamlit run main.py
```

Make sure to configure your `.streamlit/secrets.toml` file with the necessary keys (OpenAI, DB, etc.).

---

## 💼 Enterprise Readiness

- Modular and secure code
- Snowflake-ready backend
- Works with structured business data
- Plug-and-play insights for dashboards
- Easy to deploy with Streamlit Cloud or Docker

---

## 📂 File Structure

```bash
.
├── agentic_ai.py          # Agentic AI logic, prompt parsing, chart engine
├── chart_utils.py         # All chart styling and rendering logic
├── calc_utils.py          # Numeric operations like outliers, formatting
├── main.py                # Streamlit app entry point
├── .streamlit/
│   └── secrets.toml       # Secure secrets config
└── requirements.txt       # Python dependencies
```

---

## 🧪 Coming Soon

- 📥 CSV Upload via API
- 🧠 RAG & Knowledge Graph support
- 🔗 GPT-4 context-aware summaries
- 💬 Chat with your chart data

---

## 👤 Author

[Puspak Agasti](https://github.com/puspak2n) — Data & AI Leader | BI Engineer | Product Builder

---

## 📃 License

This project is licensed under the MIT License.
