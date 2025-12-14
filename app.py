import streamlit as st
import pandas as pd
import json

from llm import init_model, stream_chat_completion, retrieve_top_k

# ---------- Constants ----------
NATIONALITIES = ["Nigeria", "India"]
CURRENT_LOCATIONS = ["Nigeria", "India", "UK"]
TARGET_COUNTRIES = ["UK", "Canada", "USA", "Australia"]
GOALS = ["Study only", "Study → Work", "Work directly"]

LOW_BUDGET_THRESHOLD = 15000  # rough, not legal or financial advice

# Super rough annual cost placeholders (hackathon demo only, NOT advice)
# Used only to generate a “feasibility hint” for duration vs budget.
EST_ANNUAL_COST_GBP = {
    "UK": 25000,
    "Canada": 22000,
    "USA": 30000,
    "Australia": 28000,
}

# ---------- Page configuration ----------
# Must be very early in the script. :contentReference[oaicite:8]{index=8}
st.set_page_config(
    page_title="OpenPath Mobility Copilot",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_model(default_model="gpt-5-nano")  # Initialize model in session_state


# ---------- UI helpers ----------
def render_persona_card():
    st.markdown("### 👤 Persona: Student → Work")
    st.markdown(
        "- 🎓 **Goal:** Study, then convert to work route\n"
        "- 🧾 **Key constraints:** proof of funds, course duration, visa conditions\n"
        "- ⚠️ **Risks (high-level):** underfunding, wrong timeline assumptions, missing documents\n"
        "- ✅ **Best demo story:** “I can decode a scary email + plan next steps + draft outreach.”"
    )
    st.caption("Prototype only — not legal/immigration advice.")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


def _try_parse_json(s: str):
    try:
        return json.loads(_strip_code_fences(s)), None
    except Exception as e:
        return None, str(e)


def _render_doc_json(payload: dict):
    summary = payload.get("summary_bullets", [])
    checklist = payload.get("checklist", [])
    unknowns = payload.get("unknowns", [])
    links = payload.get("links_to_verify", [])
    sources_used = payload.get("sources_used", [])

    if summary:
        st.markdown("**Plain-English summary**")
        for b in summary:
            st.markdown(f"- {b}")

    if checklist:
        st.markdown("**Checklist**")
        for c in checklist:
            st.markdown(f"- [ ] {c}")

    if unknowns:
        st.markdown("**UNKNOWN (verify these)**")
        for u in unknowns:
            st.markdown(f"- {u}")

    if links:
        st.markdown("**Links to verify**")
        for ln in links:
            st.markdown(f"- {ln}")

    if sources_used:
        st.markdown("**Sources used (from KB retrieval)**")
        for su in sources_used:
            st.markdown(f"- {su}")


# ---------- Helper: build roadmap based on corridor + goal ----------
def build_roadmap(profile: dict) -> list[dict]:
    """
    Return a list of roadmap rows for the chosen corridor/goal.

    Each row has keys: Phase, When, Status / Visa, Key steps.

    Simplified and generic for a hackathon demo.
    """
    nationality = profile["nationality"]
    target_country = profile["target_country"]
    goal = profile["goal"]

    # IMPORTANT FIX:
    # timeline_years now means "duration in target country" (not "time until you leave").
    duration_years = profile.get("timeline_years", 0)

    rows: list[dict] = []

    # -------- UK routes --------
    if target_country == "UK":
        if goal == "Study → Work":
            rows = [
                {
                    "Phase": "Admissions & Funding",
                    "When": "Now → offer deadline",
                    "Status / Visa": "Pre-visa prep",
                    "Key steps": (
                        "Choose course and university; check eligibility; "
                        "plan proof of funds; prepare documents (passport, transcripts, etc.)."
                    ),
                },
                {
                    "Phase": "Student Route",
                    "When": f"During degree (~{duration_years} year(s))" if duration_years else "During degree",
                    "Status / Visa": "Student visa",
                    "Key steps": (
                        "Receive CAS; apply for visa; pay fees / IHS; "
                        "travel; enrol; keep attendance and financial records."
                    ),
                },
                {
                    "Phase": "Post-study work",
                    "When": "After graduation (typically up to 2 years)",
                    "Status / Visa": "Graduate Route",
                    "Key steps": (
                        "Apply inside the UK; meet eligibility; "
                        "gain UK work experience; target sponsor-licenced employers."
                    ),
                },
                {
                    "Phase": "Long-term work",
                    "When": f"After ~{duration_years + 2} years (approx.)" if duration_years else "Later (approx.)",
                    "Status / Visa": "Skilled Worker (example)",
                    "Key steps": (
                        "Secure job with sponsor-licenced employer; "
                        "meet salary/occupation rules; plan for long-term stay."
                    ),
                },
            ]
        elif goal == "Study only":
            rows = [
                {
                    "Phase": "Admissions",
                    "When": "Now → offer",
                    "Status / Visa": "Pre-visa prep",
                    "Key steps": (
                        "Shortlist universities; check entry requirements; "
                        "prepare documents; apply before deadlines."
                    ),
                },
                {
                    "Phase": "Student Route",
                    "When": f"Whole course (~{duration_years} year(s))" if duration_years else "Whole course",
                    "Status / Visa": "Student visa",
                    "Key steps": (
                        "Show proof of funds; receive CAS; apply for visa; "
                        "travel; enrol; follow visa conditions."
                    ),
                },
            ]
        elif goal == "Work directly":
            rows = [
                {
                    "Phase": "Skill & Portfolio Build",
                    "When": "0–12 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Build in-demand skills and experience; "
                        "prepare CV, portfolio, references; research UK roles."
                    ),
                },
                {
                    "Phase": "Overseas job search",
                    "When": "6–18 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Target sponsor-licenced employers; apply; "
                        "prepare for technical and behavioural interviews."
                    ),
                },
                {
                    "Phase": "Move to UK",
                    "When": "After job offer",
                    "Status / Visa": "Skilled Worker (example)",
                    "Key steps": (
                        "Employer issues Certificate of Sponsorship; "
                        "apply for visa; arrange travel and accommodation."
                    ),
                },
            ]
        else:
            rows = [
                {
                    "Phase": "Research & Plan (UK)",
                    "When": "Now → 3 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Compare UK routes (study, work, other); "
                        "check official GOV.UK guidance for your situation."
                    ),
                }
            ]

    # -------- Canada routes --------
    elif target_country == "Canada":
        if goal == "Study → Work":
            rows = [
                {
                    "Phase": "Admissions & Study Plan",
                    "When": "Now → offer",
                    "Status / Visa": "Pre-permit prep",
                    "Key steps": (
                        "Choose Designated Learning Institution (DLI); "
                        "apply; plan proof of funds and documents."
                    ),
                },
                {
                    "Phase": "Study permit",
                    "When": f"During degree (~{duration_years} year(s))" if duration_years else "During degree",
                    "Status / Visa": "Study permit",
                    "Key steps": (
                        "Apply for study permit; travel; attend classes; "
                        "respect work limits if part-time work is allowed."
                    ),
                },
                {
                    "Phase": "Post-graduation work",
                    "When": "After graduation (where eligible)",
                    "Status / Visa": "PGWP (example)",
                    "Key steps": (
                        "Apply for Post-Graduation Work Permit; "
                        "gain skilled Canadian work experience."
                    ),
                },
                {
                    "Phase": "Longer-term options",
                    "When": f"After ~{duration_years + 2} years (approx.)" if duration_years else "Later (approx.)",
                    "Status / Visa": "PR pathways (examples)",
                    "Key steps": (
                        "Explore Express Entry and provincial programs; "
                        "check current eligibility rules and points requirements."
                    ),
                },
            ]
        elif goal == "Study only":
            rows = [
                {
                    "Phase": "Admissions & Planning",
                    "When": "Now → offer",
                    "Status / Visa": "Pre-permit prep",
                    "Key steps": (
                        "Choose DLI; check language and academic requirements; "
                        "apply; plan finances and documents."
                    ),
                },
                {
                    "Phase": "Study permit",
                    "When": f"During program (~{duration_years} year(s))" if duration_years else "During program",
                    "Status / Visa": "Study permit",
                    "Key steps": (
                        "Apply for permit; complete biometrics; travel; "
                        "study; comply with permit conditions."
                    ),
                },
            ]
        elif goal == "Work directly":
            rows = [
                {
                    "Phase": "Eligibility & Skills check",
                    "When": "0–6 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Check Canadian work/immigration options relevant to your profile; "
                        "build skills and language test scores where needed."
                    ),
                },
                {
                    "Phase": "Job search & offer",
                    "When": "6–18 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Target employers and programs that fit your skills; "
                        "apply; interview; secure an offer where required."
                    ),
                },
                {
                    "Phase": "Work / immigration application",
                    "When": "Before departure",
                    "Status / Visa": "Work permit / PR (examples)",
                    "Key steps": (
                        "Follow official Canadian guidance for your chosen route; "
                        "submit documents; complete biometrics; plan travel."
                    ),
                },
            ]
        else:
            rows = [
                {
                    "Phase": "Research & Plan (Canada)",
                    "When": "Now → 3 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Compare options such as study permits, work permits, and PR pathways; "
                        "use official Canadian government resources."
                    ),
                }
            ]

    # -------- USA routes --------
    elif target_country == "USA":
        if goal == "Study → Work":
            rows = [
                {
                    "Phase": "Admissions",
                    "When": "Now → offer",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Apply to SEVP-approved schools; receive I-20; "
                        "plan finances and documents."
                    ),
                },
                {
                    "Phase": "Student journey",
                    "When": f"During degree (~{duration_years} year(s))" if duration_years else "During degree",
                    "Status / Visa": "F-1 (example)",
                    "Key steps": (
                        "Apply for F-1 visa; travel; maintain status; "
                        "follow any work limitations (on-campus, CPT where allowed)."
                    ),
                },
                {
                    "Phase": "Post-study training",
                    "When": "After graduation",
                    "Status / Visa": "OPT / CPT (where eligible)",
                    "Key steps": (
                        "Apply for practical training; gain US work experience; "
                        "discuss long-term options with employer/immigration advisor."
                    ),
                },
            ]
        elif goal == "Study only":
            rows = [
                {
                    "Phase": "Admissions",
                    "When": "Now → offer",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Apply to suitable US institutions; receive I-20; "
                        "plan funding and documents."
                    ),
                },
                {
                    "Phase": "Student journey",
                    "When": f"During degree (~{duration_years} year(s))" if duration_years else "During degree",
                    "Status / Visa": "F-1 (example)",
                    "Key steps": (
                        "Apply for visa; maintain status; complete program; "
                        "decide whether to return home or explore further options."
                    ),
                },
            ]
        elif goal == "Work directly":
            rows = [
                {
                    "Phase": "Skill & Employer targeting",
                    "When": "0–12 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Develop skills employers need; prepare CV/portfolio; "
                        "identify employers that sponsor US work visas."
                    ),
                },
                {
                    "Phase": "Job offer & petition",
                    "When": "After job offer",
                    "Status / Visa": "H-1B or other (examples)",
                    "Key steps": (
                        "Employer files petition where relevant; "
                        "complete consular processing; plan travel and relocation."
                    ),
                },
            ]
        else:
            rows = [
                {
                    "Phase": "Research US options",
                    "When": "Now → 3 months",
                    "Status / Visa": "N/A",
                    "Key steps": (
                        "Compare study, work, and other US routes; "
                        "use official US government resources."
                    ),
                }
            ]

    # -------- Australia (simple) --------
    elif target_country == "Australia":
        rows = [
            {
                "Phase": "Research & Eligibility (Australia)",
                "When": "Now → 3 months",
                "Status / Visa": "N/A",
                "Key steps": (
                    "Check Australian study or work options that match your goal; "
                    "use official Australian government guidance."
                ),
            },
            {
                "Phase": "Apply & prepare",
                "When": "3–12 months",
                "Status / Visa": "Example visa/permit",
                "Key steps": (
                    "Apply to institutions/employers; prepare finances and documents; "
                    "follow official instructions for your chosen route."
                ),
            },
        ]

    else:
        rows = [
            {
                "Phase": f"Research & Plan ({target_country})",
                "When": "Now → 3 months",
                "Status / Visa": "N/A",
                "Key steps": (
                    f"Understand {target_country} visa and study/work options using official resources."
                ),
            },
            {
                "Phase": "Apply",
                "When": "3–12 months",
                "Status / Visa": "N/A",
                "Key steps": ("Apply to institutions or employers; prepare documents and funds."),
            },
        ]

    return rows


# ---------- Sidebar navigation (your choice) ----------
st.sidebar.title("🧭 OpenPath Navigation")
nav = st.sidebar.radio(
    "Go to",
    ["🏁 Demo mode", "🧙 Wizard → Roadmap", "📄 Doc explainer", "🤝 Collab helper", "🧩 Show all tabs"],
    key="nav",
)
with st.sidebar.expander("👤 Persona card"):
    render_persona_card()

st.sidebar.markdown("---")
st.sidebar.caption("Hackathon prototype • not legal advice")


# ---------- Main header ----------
st.title("OpenPath – Mobility Copilot")
st.write("Turn visa chaos into a clear roadmap. **Prototype – not legal or immigration advice.**")


# Keep tabs (so you’re not “removing” your existing structure). :contentReference[oaicite:9]{index=9}
tab_wizard, tab_doc, tab_collab = st.tabs(["Wizard → Roadmap", "Doc explainer", "Collab helper"])


# ====================== DEMO MODE (Landing page) ======================
if nav == "🏁 Demo mode":
    st.markdown("## 🏁 Demo mode (Judge-friendly story)")
    st.markdown(
        "### One sentence\n"
        "**OpenPath** turns confusing visa + university requirements into a clear, action-focused plan.\n\n"
        "### What you can demo in 60 seconds\n"
        "1) Build a corridor roadmap (Wizard)\n"
        "2) Paste a scary email → get a checklist + UNKNOWNs (Doc explainer + RAG)\n"
        "3) Draft a professional email + time slots (Collab helper)\n"
    )

    st.markdown("### Why it’s innovative for VisaVerse")
    st.markdown(
        "- 🌍 Helps people move across borders with less confusion\n"
        "- 🧠 Uses AI + retrieval (RAG) for transparency\n"
        "- 🤝 Supports collaboration (emails + time zones)\n"
    )

    st.markdown("### Persona spotlight")
    render_persona_card()

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("🚀 Start Wizard", use_container_width=True):
            st.session_state["nav"] = "🧙 Wizard → Roadmap"
            st.rerun()
    with colB:
        st.info("Tip for judging: after Wizard, paste an email in Doc explainer to show RAG sources.")


# ====================== WIZARD TAB ======================
with tab_wizard:
    if nav not in ["🧙 Wizard → Roadmap", "🧩 Show all tabs"]:
        st.info("Use the sidebar to open **Wizard → Roadmap**.")
    else:
        st.subheader("Step 1 – Tell us about your journey")

        # Progress stepper (simple UX win). :contentReference[oaicite:10]{index=10}
        if "wizard_step" not in st.session_state:
            st.session_state.wizard_step = 1
        st.progress(st.session_state.wizard_step / 3, text=f"Step {st.session_state.wizard_step} of 3")

        col1, col2 = st.columns(2)
        with col1:
            nationality = st.selectbox("Your nationality", NATIONALITIES)
            current_country = st.selectbox("Where are you now?", CURRENT_LOCATIONS)
        with col2:
            target_country = st.selectbox("Where do you want to go?", TARGET_COUNTRIES)
            goal = st.selectbox("Goal", GOALS)

        # NEW: Start timing vs Duration timing (fixes your flaw)
        start_in_years = st.slider(
            "When do you want to start this journey? (years from now)",
            min_value=0,
            max_value=5,
            step=1,
            help="This is the time until you begin your journey.",
        )

        # Duration (what you said was missing / confusing before)
        duration_years = st.slider(
            "How long will you be in the target country? (years)",
            min_value=1,
            max_value=6,
            step=1,
            help="This is your planned time *in* the target country (e.g., program duration).",
        )

        budget = st.number_input(
            "Approximate total budget (£)",
            min_value=0,
            step=500,
            help="Very rough idea: tuition + living + visa/health + travel.",
        )

        st.caption(
            f"🗓️ **Start in:** {start_in_years} year(s) • "
            f"⏳ **Planned duration in {target_country}:** {duration_years} year(s)"
        )

        # Feasibility hint (demo-only)
        annual = EST_ANNUAL_COST_GBP.get(target_country)
        if annual and budget > 0:
            est_required = annual * duration_years
            gap = budget - est_required

            m1, m2, m3 = st.columns(3)
            m1.metric("Est. annual cost (rough)", f"£{annual:,}")
            m2.metric("Est. required budget", f"£{est_required:,}")
            m3.metric("Budget gap", f"£{gap:,}")

            if budget < est_required:
                st.warning(
                    "Your budget may not cover the full planned duration (rough estimate). "
                    "Consider scholarships, cheaper regions, shorter programs, or phased plans. "
                    "Always verify real costs from official sources."
                )

        # -------- Initialize session_state --------
        if "roadmap_history" not in st.session_state:
            st.session_state.roadmap_history = []

        if "selected_corridor" not in st.session_state:
            st.session_state.selected_corridor = None

        # -------- Button: lock & show roadmap --------
        if st.button("Lock this corridor and build roadmap"):
            corridor = f"{nationality} → {target_country}"
            st.session_state.selected_corridor = corridor

            profile = {
                "nationality": nationality,
                "current_country": current_country,
                "target_country": target_country,
                "goal": goal,
                "budget": budget,
                # IMPORTANT: keep your original key name but now it means duration in target country
                "timeline_years": duration_years,
                "start_in_years": start_in_years,
            }
            st.session_state.roadmap_history.append(profile)

            rows = build_roadmap(profile)
            df = pd.DataFrame(rows)

            st.session_state.wizard_step = 2  # stepper UX
            st.progress(st.session_state.wizard_step / 3, text=f"Step {st.session_state.wizard_step} of 3")

            st.subheader("Roadmap overview")
            st.write(f"Your roadmap has **{len(rows)}** main phase(s).")

            st.table(df)  # static
            st.dataframe(df, use_container_width=True)  # interactive

            status_counts = df["Status / Visa"].value_counts().reset_index()
            status_counts.columns = ["Status / Visa", "Count"]

            st.metric("Number of phases", len(rows))
            st.bar_chart(status_counts.set_index("Status / Visa"))

            if budget > 0 and budget < LOW_BUDGET_THRESHOLD:
                st.warning(
                    "Your budget looks relatively low for an international move. "
                    "You may need scholarships, cheaper locations, or a phased approach. "
                    "Always verify actual costs with official and trusted sources."
                )

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download roadmap as CSV",
                csv,
                "openpath_roadmap.csv",
                "text/csv",
            )

            st.session_state.wizard_step = 3
            st.progress(st.session_state.wizard_step / 3, text="Step 3 of 3 • Roadmap ready ✅")

            st.caption("⚠️ Simplified guide — always verify on official government/university websites.")

        # Snapshot
        if st.session_state.selected_corridor and st.session_state.roadmap_history:
            st.markdown("---")
            st.markdown("#### Current corridor snapshot")
            latest = st.session_state.roadmap_history[-1]
            st.write(
                f"**Corridor:** {latest['nationality']} → {latest['target_country']}  |  "
                f"**Goal:** {latest['goal']}  |  "
                f"**Budget:** ~£{latest['budget']:,}  |  "
                f"**Start in:** {latest.get('start_in_years', 0)} year(s)  |  "
                f"**Duration in-country:** {latest['timeline_years']} year(s)"
            )


# ====================== DOC EXPLAINER TAB ======================
with tab_doc:
    if nav not in ["📄 Doc explainer", "🧩 Show all tabs"]:
        st.info("Use the sidebar to open **Doc explainer**.")
    else:
        st.subheader("Doc explainer – make confusing emails simple")
        st.caption(
            "Paste visa or university emails/requirements below. "
            "OpenPath will rewrite them in plain English with a checklist. "
            "This is guidance only, not legal or immigration advice."
        )

        if "doc_messages" not in st.session_state:
            st.session_state.doc_messages = []

        if "last_doc_snips" not in st.session_state:
            st.session_state.last_doc_snips = []

        # Render history
        for message in st.session_state.doc_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    parsed, _err = _try_parse_json(message["content"])
                    if parsed:
                        _render_doc_json(parsed)
                        with st.expander("Raw JSON"):
                            st.json(parsed)
                    else:
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])

        if prompt := st.chat_input(
            "Paste an email/requirement or ask a visa question...",
            key="doc_chat_input",
        ):
            st.session_state.doc_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            snips = retrieve_top_k(prompt, k=3)
            st.session_state.last_doc_snips = snips

            if snips:
                sources_block = "SOURCES (use only if relevant; if detail missing, say UNKNOWN):\n"
                for i, s in enumerate(snips, start=1):
                    sources_block += (
                        f"[{i}] {s.title} | file={s.kb_file} | url={s.source_url}\n"
                        f"{s.text}\n\n"
                    )
            else:
                sources_block = "SOURCES: (none)\n"

            system_msg = {
                "role": "system",
                "content": (
                    "You explain immigration/university emails in very simple English.\n"
                    "You are NOT a lawyer. This is NOT legal/immigration advice.\n\n"
                    "RULES:\n"
                    "- Use SOURCES if relevant.\n"
                    "- If a detail is NOT in SOURCES, write UNKNOWN and tell the user what to verify.\n"
                    "- Do NOT invent fees, dates, eligibility rules.\n"
                    "- Output MUST be valid JSON ONLY (no markdown, no code fences) with keys:\n"
                    "  summary_bullets, checklist, unknowns, links_to_verify, sources_used\n\n"
                    f"{sources_block}"
                ),
            }
            messages = [system_msg] + st.session_state.doc_messages

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for chunk in stream_chat_completion(messages):
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_response += delta
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                parsed, err = _try_parse_json(full_response)
                if parsed:
                    st.markdown("---")
                    _render_doc_json(parsed)
                    with st.expander("Raw JSON"):
                        st.json(parsed)
                else:
                    st.warning(
                        "Model output was not valid JSON. Showing raw output instead.\n\n"
                        f"Parse error: {err}"
                    )
                    st.code(full_response)

            st.session_state.doc_messages.append({"role": "assistant", "content": full_response})

        # ✅ Sources viewer dropdown (judge transparency)
        with st.expander("🔎 Sources viewer (dropdown)"):
            snips = st.session_state.last_doc_snips or []
            if not snips:
                st.caption("No retrieved sources yet. Ask a question above first.")
            else:
                options = [f"{s.title} — {s.kb_file}" for s in snips]
                pick = st.selectbox("Choose a retrieved chunk to view", options, key="source_pick")
                chosen = snips[options.index(pick)]
                st.markdown(f"**Title:** {chosen.title}")
                st.markdown(f"**File:** `{chosen.kb_file}`")
                st.markdown(f"**URL:** {chosen.source_url or 'UNKNOWN'}")
                st.text_area("Chunk text", chosen.text, height=220)


# ====================== COLLAB HELPER TAB ======================
with tab_collab:
    if nav not in ["🤝 Collab helper", "🧩 Show all tabs"]:
        st.info("Use the sidebar to open **Collab helper**.")
    else:
        st.subheader("Collab helper – emails & cross-time-zone planning")
        st.caption(
            "Describe who you want to contact (university, employer, agent), "
            "what you want (meeting, clarification, etc.), and any time zones. "
            "OpenPath will draft a polite email/message and suggest a few time slots."
        )

        if "collab_messages" not in st.session_state:
            st.session_state.collab_messages = []

        for message in st.session_state.collab_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(
            "Who are you contacting and what do you want to say?",
            key="collab_chat_input",
        ):
            st.session_state.collab_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You help international students write short, polite emails/messages "
                        "to universities, employers, or agents.\n"
                        "If they mention a meeting/call, suggest 2–3 specific time slots and "
                        "explicitly mention time zones.\n"
                        "Keep the message clean, professional, and easy to copy-paste."
                    ),
                },
                *st.session_state.collab_messages,
            ]

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for chunk in stream_chat_completion(messages):
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_response += delta
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            st.session_state.collab_messages.append({"role": "assistant", "content": full_response})


# ====================== Optional multipage note (no code change needed now) ======================
# If your app grows, Streamlit supports true multipage navigation via `pages/` or st.navigation/st.Page,
# and you can programmatically route with st.switch_page. :contentReference[oaicite:11]{index=11}
