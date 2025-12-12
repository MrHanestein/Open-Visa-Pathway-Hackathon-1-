import streamlit as st
import pandas as pd

# ---------- Constants ----------
NATIONALITIES = ["Nigeria", "India"]
CURRENT_LOCATIONS = ["Nigeria", "India", "UK"]
TARGET_COUNTRIES = ["UK", "Canada", "USA", "Australia"]
GOALS = ["Study only", "Study → Work", "Work directly"]

LOW_BUDGET_THRESHOLD = 15000  # rough, not legal or financial advice

# ---------- Page configuration ----------
st.set_page_config(page_title="OpenPath Mobility Copilot", page_icon="🌍")

st.title("OpenPath – Mobility Copilot")
st.write(
    "Turn visa chaos into a clear roadmap. "
    "**Prototype – not legal or immigration advice.**"
)

tab_wizard, tab_doc, tab_collab = st.tabs(
    ["Wizard → Roadmap", "Doc explainer", "Collab helper"]
)


# ---------- Helper: build roadmap based on corridor + goal ----------
def build_roadmap(profile: dict) -> list[dict]:
    """
    Return a list of roadmap rows for the chosen corridor/goal.

    Each row has keys: Phase, When, Status / Visa, Key steps.

    This is deliberately simplified and generic for a hackathon demo.
    Always double-check with official government / university sources.
    """

    nationality = profile["nationality"]
    target_country = profile["target_country"]
    goal = profile["goal"]
    timeline_years = profile["timeline_years"]

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
                    "When": "During degree",
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
                    "When": f"{timeline_years + 3} years onward (approx.)",
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
                    "When": "Whole course",
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
                    "When": "During degree",
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
                    "When": f"{timeline_years + 3} years onward (approx.)",
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
                    "When": "During program",
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
                    "When": "During degree",
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
                    "When": "During degree",
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

    # -------- Example extra target: Australia (simple) --------
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

    # -------- Fallback generic (currently rarely used) --------
    else:
        rows = [
            {
                "Phase": f"Research & Plan ({target_country})",
                "When": "Now → 3 months",
                "Status / Visa": "N/A",
                "Key steps": (
                    f"Understand {target_country} visa and study/work options using "
                    "official government resources."
                ),
            },
            {
                "Phase": "Apply",
                "When": "3–12 months",
                "Status / Visa": "N/A",
                "Key steps": (
                    "Apply to institutions or employers; prepare documents and funds."
                ),
            },
        ]

    return rows


# ====================== WIZARD TAB ======================
with tab_wizard:
    st.subheader("Step 1 – Tell us about your journey")

    col1, col2 = st.columns(2)
    with col1:
        nationality = st.selectbox("Your nationality", NATIONALITIES)
        current_country = st.selectbox("Where are you now?", CURRENT_LOCATIONS)
    with col2:
        target_country = st.selectbox("Where do you want to go?", TARGET_COUNTRIES)
        goal = st.selectbox("Goal", GOALS)

    budget = st.number_input(
        "Approximate total budget (£)",
        min_value=0,
        step=500,
        help="Very rough idea: tuition + living + visa/health + travel.",
    )
    timeline_years = st.slider(
        "When do you want to move?",
        min_value=0,
        max_value=5,
        step=1,
        help="Select how many years from now you aim to start this journey.",
    )

    st.caption(f"⏱️ Selected timeline: about **{timeline_years} year(s)** from now.")

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
            "timeline_years": timeline_years,
        }
        st.session_state.roadmap_history.append(profile)

        rows = build_roadmap(profile)
        df = pd.DataFrame(rows)

        st.subheader("Roadmap overview")
        st.write(f"Your roadmap has **{len(rows)}** main phase(s).")
        st.dataframe(df, use_container_width=True)

        # Budget-based gentle warning (not advice)
        if budget > 0 and budget < LOW_BUDGET_THRESHOLD:
            st.warning(
                "Your budget looks relatively low for an international move. "
                "You may need scholarships, cheaper locations, or a phased approach. "
                "Always verify actual costs with official and trusted sources."
            )

        # Optional: CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download roadmap as CSV",
            csv,
            "openpath_roadmap.csv",
            "text/csv",
        )

        st.caption(
            "⚠️ This is a simplified guide. "
            "Always double-check policies on official government and university websites."
        )

    # Show current snapshot even after rerun
    if st.session_state.selected_corridor and st.session_state.roadmap_history:
        st.markdown("---")
        st.markdown("#### Current corridor snapshot")
        latest = st.session_state.roadmap_history[-1]
        st.write(
            f"**Corridor:** {latest['nationality']} → {latest['target_country']}  |  "
            f"**Goal:** {latest['goal']}  |  "
            f"**Budget:** ~£{latest['budget']:,}  |  "
            f"**Timeline:** {latest['timeline_years']} year(s)"
        )

# ====================== DOC EXPLAINER TAB (to be wired with OpenAI) ======================
with tab_doc:
    st.subheader("Doc explainer")
    st.info(
        "Paste visa or university emails/requirements here (feature to be wired to AI)."
    )

# ====================== COLLAB HELPER TAB (future) ======================
with tab_collab:
    st.subheader("Collab helper")
    st.info(
        "This tab will help you schedule across time zones and draft emails/messages "
        "for universities, agents, or employers (future version)."
    )
