import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import time
import base64

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# Environment & OpenAI client
# -------------------------------------------------
load_dotenv()  # will read OPENAI_API_KEY from .env if present
client = OpenAI()  # uses OPENAI_API_KEY from env by default

BASE_DIR = Path(__file__).parent  # for robust asset loading

# -------------------------------------------------
# Data class for the judge result
# -------------------------------------------------
@dataclass
class Judgment:
    neutral_summary: str
    a_responsibility: int
    b_responsibility: int
    advice_a: str
    advice_b: str
    apology_template: str
    safety_flag: bool = False
    safety_message: str = ""


# -------------------------------------------------
# Fallback / demo logic (used if LLM fails)
# -------------------------------------------------
def get_judgment_mock(story_a: str, story_b: str, tone: str) -> Judgment:
    """
    Simple heuristic used as a fallback or placeholder:
    splits responsibility roughly based on how long each story is.
    """

    len_a = len(story_a)
    len_b = len(story_b)
    total = max(len_a + len_b, 1)
    a_share = int(round(len_a / total * 100))
    b_share = 100 - a_share

    neutral_summary = (
        "From both perspectives, this looks like a mix of unmet expectations "
        "and communication gaps rather than one person being purely right or wrong. "
        "Both people had reasons for what they did, but those reasons weren’t clearly shared."
    )

    advice_a = (
        "Try to name what you needed earlier and out loud. Instead of waiting and "
        "hoping they guess, say something like: “This is important to me because…”. "
        "That gives them a fair chance to respond."
    )

    advice_b = (
        "Acknowledge the impact of your actions, even if you didn’t mean harm. "
        "You can say: “I see how that hurt you, even though I didn’t intend it.” "
        "Then share a bit of your own constraints calmly."
    )

    apology_template = (
        "Hey, I’ve been thinking about what happened. I’m sorry for the part I played "
        "in how things went. I didn’t mean to make you feel that way. Next time, I’ll "
        "try to be more clear about what I’m thinking and I’ll check in with you sooner "
        "instead of letting the tension build up."
    )

    return Judgment(
        neutral_summary=neutral_summary,
        a_responsibility=a_share,
        b_responsibility=b_share,
        advice_a=advice_a,
        advice_b=advice_b,
        apology_template=apology_template,
        safety_flag=False,
        safety_message=""
    )


# -------------------------------------------------
# LLM-powered judgment
# -------------------------------------------------
def get_judgment_llm(story_a: str, story_b: str, tone: str) -> Judgment:
    """
    Calls the OpenAI Chat Completions API and asks for a JSON object
    with all fields needed for the UI.
    We force JSON output via response_format={"type": "json_object"}.
    """

    system_prompt = """
You are an empathetic but honest conflict mediator, represented as a cute owl judge.
You will be given two perspectives (Person A and Person B) about the same situation.

Your job is to:
1. Summarize the situation in a NEUTRAL, non-judgmental way.
2. Assign a percentage of responsibility to each person (a_responsibility and b_responsibility),
   such that they are integers that add up to 100.
3. Give concrete, practical advice for what Person A could do differently in the future.
4. Give concrete, practical advice for what Person B could do differently in the future.
5. Provide a short apology template that either person could use as a starting point.

SAFETY:
- If the situation involves abuse, severe harassment, self-harm, suicidal ideation,
  or anything that requires professional help, set safety_flag to true and
  set safety_message to a brief, kind suggestion to seek real-world support.
- In such cases, avoid blaming a victim. Focus on safety and support, not blame.

FORMAT:
Respond with a single JSON object with this schema:

{
  "neutral_summary": "string",
  "a_responsibility": 0,
  "b_responsibility": 0,
  "advice_a": "string",
  "advice_b": "string",
  "apology_template": "string",
  "safety_flag": false,
  "safety_message": "string"
}
    """.strip()

    user_prompt = f"""
Tone: {tone}

Person A story:
\"\"\"{story_a}\"\"\"


Person B story:
\"\"\"{story_b}\"\"\"    
    """.strip()

    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw_text = resp.choices[0].message.content.strip()
    print("RAW LLM OUTPUT:", raw_text)

    data = json.loads(raw_text)

    a_resp = int(data.get("a_responsibility", 50))
    b_resp = int(data.get("b_responsibility", 50))

    # normalize to sum to 100 in case the model is slightly off
    total = a_resp + b_resp or 1
    a_resp = int(round(a_resp / total * 100))
    b_resp = 100 - a_resp

    return Judgment(
        neutral_summary=data.get("neutral_summary", "").strip(),
        a_responsibility=a_resp,
        b_responsibility=b_resp,
        advice_a=data.get("advice_a", "").strip(),
        advice_b=data.get("advice_b", "").strip(),
        apology_template=data.get("apology_template", "").strip(),
        safety_flag=bool(data.get("safety_flag", False)),
        safety_message=data.get("safety_message", "").strip(),
    )


def get_judgment(story_a: str, story_b: str, tone: str) -> Judgment:
    """
    Wrapper that tries the LLM first and falls back to the mock logic
    if anything goes wrong (JSON error, network issue, auth error, etc.).
    """
    try:
        return get_judgment_llm(story_a, story_b, tone)
    except Exception as e:
        print("LLM call failed, using mock judgment instead:", repr(e))
        return get_judgment_mock(story_a, story_b, tone)


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Friendship Court",
    page_icon="🦉",
    layout="centered"
)
app_bg_path = BASE_DIR / "assets" / "app_background.jpg"

if app_bg_path.exists():
    app_bg_bytes = app_bg_path.read_bytes()
    app_bg_b64 = base64.b64encode(app_bg_bytes).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{app_bg_b64}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- Session State ----------------
if "phase" not in st.session_state:
    # "input" → show form
    # "thinking" → show video while LLM runs
    # "results" → show step 1–3
    st.session_state.phase = "input"

if "step" not in st.session_state:
    st.session_state.step = 0  # 0 only used inside "input" phase

if "judgment" not in st.session_state:
    st.session_state.judgment: Optional[Judgment] = None

def next_step():
    st.session_state.step = min(3, st.session_state.step + 1)

def prev_step():
    st.session_state.step = max(1, st.session_state.step - 1)

def reset_case():
    st.session_state.phase = "input"
    st.session_state.step = 0
    st.session_state.judgment = None
    st.session_state.pop("pending_story_a", None)
    st.session_state.pop("pending_story_b", None)
    st.session_state.pop("pending_tone", None)




# -------------------------------------------------
# PHASE: THINKING (show ONLY the video, then run LLM)
# -------------------------------------------------
if st.session_state.phase == "thinking":
    video_path = BASE_DIR / "assets" / "loading.mov"  # change name if needed

    if video_path.exists():
        video_bytes = video_path.read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        # Full-screen centered autoplay video
        video_html = f"""
        <div style="
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            height:100vh;
            background-color:#eaeaea;
        ">
          <h2 style="color:#2D3142; font-family:system-ui; margin-bottom:16px;">
            The owl is thinking…
          </h2>
          <video autoplay playsinline width="480">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.info("Thinking animation video not found. Put it at assets/loading.mov")
        # Fallback spinner if video missing
        with st.spinner("The owl is thinking…"):
            time.sleep(3)

    # Run LLM + video in parallel-ish: LLM first, then wait for remaining video time
    start = time.time()

    # Run LLM using stored stories and tone
    story_a = st.session_state.get("pending_story_a", "")
    story_b = st.session_state.get("pending_story_b", "")
    tone = st.session_state.get("pending_tone", "Gentle")

    judgment = get_judgment(story_a, story_b, tone)
    st.session_state.judgment = judgment

    # How long did the LLM take?
    elapsed = time.time() - start
    video_duration = 10  # seconds – change if your video length changes
    remaining = max(0, video_duration - elapsed)

    # If LLM finished early, wait until video is done
    time.sleep(remaining)

    # Switch to results phase step 1 (verdict)
    st.session_state.phase = "results"
    st.session_state.step = 1

    st.rerun()





# ---------------- CSS Theme ----------------
st.markdown(
    """
    <style>
    .stApp {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }

    .fc-center {
        text-align: center;
    }

    .fc-subtle {
        color: #6a6d78;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    .fc-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        background: #f3f5ff;
        border: 1px solid #dfe3ff;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #5560ff;
        margin-bottom: 0.4rem;
    }

    .fc-pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(135deg, #ffd4d8, #ffe5e7);
        box-shadow: 0 0 6px rgba(255, 154, 162, 0.5);
    }

    /* 💡 NEW CLEAN CARD STYLE */
    .fc-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e8e9ef;
        padding: 18px 18px 14px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.05);
        margin-bottom: 14px;
    }

    .fc-card h3 {
        margin: 0 0 0.35rem 0;
        font-size: 1.05rem;
        color: #2d3142;
    }

    .fc-card p {
        margin: 0;
        font-size: 0.9rem;
        color: #3c4156;
        line-height: 1.55;
    }

    .fc-card-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #8a8ea8;
        margin-bottom: 0.4rem;
    }

    .fc-caption {
        font-size: 0.75rem;
        color: #8a8ea8;
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- Header ----------------
st.markdown('<div class="fc-center">', unsafe_allow_html=True)

st.markdown(
    '<div class="fc-pill"><div class="fc-pill-dot"></div><span>LLM Conflict Mediator</span></div>',
    unsafe_allow_html=True
)

owl_path = BASE_DIR / "assets" / "owl_judge.png"
if owl_path.exists():
    st.image(str(owl_path), width=190)
else:
    st.caption("Tip: put `owl_judge.png` inside an `assets` folder next to this file.")

st.markdown("## Friendship Court 🦉⚖️", unsafe_allow_html=True)
st.markdown(
    '<p class="fc-subtle">Two perspectives enter. One very patient owl helps you see the conflict more clearly.</p>',
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")  # spacer

# =====================================================
# STEP 0: INPUT FORM
# =====================================================
if st.session_state.step == 0:
    with st.form(key="friendship_form"):
        st.markdown("#### Tell the owl what happened")

        col_a, col_b = st.columns(2)

        with col_a:
            story_a = st.text_area(
                "Person A — your perspective",
                height=200,
                placeholder="What happened, how did you feel, and what did you wish had happened?"
            )
        with col_b:
            story_b = st.text_area(
                "Person B — their perspective",
                height=200,
                placeholder="Describe how you think they see the situation."
            )

        col1, col2 = st.columns([2, 1])
        with col1:
            tone = st.selectbox(
                "Judge style",
                ["Gentle", "Neutral", "Direct"],
                help="This changes how the owl talks, not the fairness of the verdict."
            )
        with col2:
            submit = st.form_submit_button("Ask the Judge 🦉")

    if submit:
        if not story_a or not story_b:
            st.warning("Please enter both perspectives so the owl has something to think about.")
        else:
            # Save inputs for the thinking phase
            st.session_state.pending_story_a = story_a
            st.session_state.pending_story_b = story_b
            st.session_state.pending_tone = tone

            # Switch to thinking phase (video-only screen)
            st.session_state.phase = "thinking"
            st.rerun()

# =====================================================
# STEPS 1–3: RESULTS FLOW
# =====================================================
if st.session_state.step >= 1 and st.session_state.judgment is not None:
    judgment: Judgment = st.session_state.judgment

    st.divider()
    top_cols = st.columns([1, 1])
    with top_cols[0]:
        st.button("🔄 Start a new case", key="reset_top", on_click=reset_case)
    with top_cols[1]:
        st.progress(st.session_state.step / 3)
        st.caption(f"Step {st.session_state.step} of 3")

    if judgment.safety_flag:
        st.error(judgment.safety_message)

    # ---------------- STEP 1: VERDICT / NEUTRAL SUMMARY ----------------
    if st.session_state.step == 1:
        st.markdown("### Verdict")

        st.markdown(
            f"""
            <div class="fc-card">
              <div class="fc-card-label">Neutral recap</div>
              <h3>What the owl heard</h3>
              <p>{judgment.neutral_summary}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.button("Next: Responsibility breakdown 📊", on_click=next_step)

    # ---------------- STEP 2: RESPONSIBILITY + CHART ----------------
    if st.session_state.step == 2:
        st.markdown("### Responsibility Breakdown")

        st.markdown(
            """
            <div class="fc-card">
              <div class="fc-card-label">Responsibility</div>
              <h3>How the owl splits the responsibility</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        chart_df = pd.DataFrame(
            {
                "Person": ["Person A", "Person B"],
                "Responsibility (%)": [
                    judgment.a_responsibility,
                    judgment.b_responsibility,
                ],
            }
        )

        chart = (
            alt.Chart(chart_df)
            .mark_bar(
                size=50,                      # thinner bars
                cornerRadiusTopLeft=12,
                cornerRadiusTopRight=12
            )
            .encode(
                x=alt.X("Person:N", title="", sort=None),
                y=alt.Y(
                    "Responsibility (%):Q",
                    title="Responsibility (%)",
                    scale=alt.Scale(domain=[0, 100])
                ),
                color=alt.Color(
                    "Person:N",
                    scale=alt.Scale(range=["#FFB3C1", "#BFDFFF"]),  # softer pastels
                    legend=None,
                ),
                tooltip=["Person", "Responsibility (%)"],
            )
            .properties(height=180, width =520)   # shorter chart
        )


        st.altair_chart(chart, use_container_width=True)

        st.write(
            f"- **Person A:** ~{judgment.a_responsibility}% responsible\n"
            f"- **Person B:** ~{judgment.b_responsibility}% responsible\n\n"
            "This isn’t about punishment, but about understanding where each person "
            "has room to act differently next time."
        )

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.button("⬅️ Back to verdict", on_click=prev_step)
        with c2:
            st.button("Next: Things you could try 💡", on_click=next_step)

    # ---------------- STEP 3: ADVICE + APOLOGY TEMPLATE ----------------
    if st.session_state.step == 3:
        st.markdown("### Things You Could Try")

        col_advice_a, col_advice_b = st.columns(2)

        with col_advice_a:
            st.markdown(
                f"""
                <div class="fc-card">
                  <div class="fc-card-label">For Person A</div>
                  <h3>Things you could try</h3>
                  <p>{judgment.advice_a}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_advice_b:
            st.markdown(
                f"""
                <div class="fc-card">
                  <div class="fc-card-label">For Person B</div>
                  <h3>Things you could try</h3>
                  <p>{judgment.advice_b}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="fc-card">
              <div class="fc-card-label">Template</div>
              <h3>Apology script you can customize</h3>
              <p>{judgment.apology_template}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="fc-caption">Friendship Court is for reflection and better conversations only. '
            'It is not a substitute for professional or medical advice.</p>',
            unsafe_allow_html=True
        )

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.button("⬅️ Back to responsibility", key="back_to_resp", on_click=prev_step)
        with c2:
            st.button("🔄 Start a new case", key="reset_bottom", on_click=reset_case)
