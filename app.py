import streamlit as st
from scoring_engine import analyze_transcript

st.set_page_config(
    page_title="Nirmaan AI – Communication Scorer",
    layout="wide",
)

# ---------- Title Section ----------
st.markdown(
    """
    <h1 style='text-align:center; color:#4A90E2;'>
        🌟 Nirmaan AI – Self Introduction Scoring Tool 🌟
    </h1>
    <p style='text-align:center; font-size:18px; color:#444;'>
        Paste your transcript or upload a text file.  
        Receive a clean, beautiful and easy-to-understand AI evaluation.
    </p>
    """,
    unsafe_allow_html=True
)

# ---------- Input Section ----------
col1, col2 = st.columns([1, 1])

with col1:
    text_input = st.text_area(
        "📝 Transcript Text",
        height=250,
        placeholder="Paste the self-introduction transcript here...",
    )

with col2:
    uploaded = st.file_uploader("📂 ...or upload a .txt file", type=["txt"])
    duration = st.number_input(
        "⏱ Approximate speaking duration (seconds)",
        min_value=10.0,
        max_value=600.0,
        value=60.0,
        step=5.0
    )

if uploaded:
    text_input = uploaded.read().decode("utf-8")


# ---------- Scoring Button ----------
if st.button("💡 Score Transcript", use_container_width=True):
    if not text_input.strip():
        st.warning("⚠ Please paste text or upload a .txt file.")
    else:
        with st.spinner("🔍 Analyzing transcript..."):
            result = analyze_transcript(text_input, speech_duration_seconds=duration)

        # ---------- Overall Score ----------
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:12px; 
                        background:linear-gradient(135deg, #6EE7B7, #3B82F6);
                        color:white; text-align:center; margin-bottom:20px;">
                <h2 style='margin:0;'>🟩 Overall Score: {result['overall_score']} / 100</h2>
                <p style='margin:0; font-size:18px;'>Total Words: {result['total_words']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------- Per-Criterion Scores ----------
        st.markdown(
            "<h3 style='color:#4A90E2;'>📊 Detailed Criterion Scores</h3>",
            unsafe_allow_html=True
        )

        for crit in result["criteria"]:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px;
                            background:#F0F7FF; margin-bottom:15px;">
                    <h4 style="color:#1F4E79; margin:0 0 10px 0;">
                        {crit['criterion']} — {crit['score']} / {crit['max_score']}
                    </h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Each Metric (Without DETAILS)
            for m in crit["metrics"]:
                st.markdown(
                    f"""
                    <div style="padding:12px; border-left: 5px solid #4A90E2; 
                                margin-bottom:10px; background:white; border-radius:8px;">
                        <p style="font-size:16px; margin:0;">
                            <strong>{m['metric']}</strong>  
                            <br>🔹 Final Score: <strong>{m['final_score']}</strong>
                            <br>🔹 Rule Score: {m['rule_score']}
                            <br>🔹 Semantic Match: {m['semantic_similarity']}
                        </p>
                        <p style="color:#333; margin-top:8px;">💬 {m['feedback']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                    # ---------- Motivational Footer ----------
        st.markdown(
            """
            <div style="margin-top:40px; padding:20px; 
                        border-radius:12px; background:#E8F1FF; 
                        border-left: 6px solid #4A90E2;">
                <h3 style="color:#1F4E79; margin-top:0;">
                    ✨ Keep Going, You're Doing Amazing!
                </h3>
                <p style="font-size:16px; color:#333;">
                    Every great speaker starts with a single introduction.  
                    Keep practicing, refining and believing in yourself,  
                    improvement is already happening with every attempt! 🌱
                    <br><br>
                    Stay confident, stay curious and keep shining bright!  
                    Your voice matters more than you know. 🌟💙  
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
