# Open-Visa-Pathway – Mobility Copilot for Global Students emigration
Website (Deployed on streamlit): https://open-visa-pathway.streamlit.app/

## 1. Problem

Nigerian and Indian international students who want to study in the UK/Canada/USA face confusing, misleading, and disorganized information
about visas, timelines, and post-study options.

## 2. Solution

Open-Visa-Pathway is an AI mobility copilot that turns this distressing process into a clear roadmap:
- Wizard → personalized roadmap (phases, checklist, risks)
- Doc explainer → plain-English summaries of visa emails/requirements
- Collab helper → draft emails/messages and time windows for cross-border calls

**Primary corridors (v1):**
- Nigeria → UK (Student → Graduate Route → Skilled Worker)
- India → UK (Student → Graduate Route → Skilled Worker)

## 3. Tech stack

- Python + Streamlit for UI
- OpenAI API (LLMs) for reasoning and generation


- (Future implementation) Chroma/FAISS for retrieval-augmented generation (RAG) alongside scaling.

## 4. Project status

- [ ] Streamlit app skeleton
- [ ] OpenAI integration
- [ ] Basic roadmap flow
- [ ] Document explainer
- [ ] Collaborator helper
## Setup (Local)
``` bash
1) git clone https://github.com/MrHanestein/Open-Visa-Pathway-Hackathon-1-.git
 cd https://github.com/MrHanestein/Open-Visa-Pathway-Hackathon-1-.git - Your local directory.
2) pip install -r requirements.txt
3) Create .streamlit/secrets.toml:
   OPENAI_API_KEY="your_key"
4) Create and activate virtual environment (Windows users): python -m venv .venv
..venv\Scripts\Activate.ps1. (MacBook users):python3 -m venv .venv
source .venv/bin/activate.
5) streamlit run app.py

Note: using .\.venv\Scripts\Activate.ps1 ensures you switch from your global Python dependencies to the isolated ones in this project.
## Theme
UI theme is configured in .streamlit/config.toml.

```


## Additional Information:
## Why my choice of Streamlit?

Open-Visa-Pathway uses Streamlit because:
- It lets me build web apps in pure Python, which is ideal for rapid hackathon prototyping.
- It has built-in widgets and layout primitives for multi-step forms and chat-like interfaces.
- It deploys easily (Streamlit Community Cloud / Railway) so judges can try the app quickly.

## Disclosure

The full code base was built during the hackathon with 10 days left on the clock.
