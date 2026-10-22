"""CareFlow AI - human-in-the-loop, multilingual, multimodal healthcare triage assistant.

Run:  streamlit run app.py
The AI never diagnoses and never makes the final decision: a qualified clinician reviews, edits and approves.
"""
import difflib
import hashlib
import html
import json
import os

import pandas as pd
import streamlit as st

import storage as db
import triage_engine as eng

st.set_page_config(page_title="CareFlow AI", page_icon="🩺", layout="wide")


@st.cache_resource
def _boot():
    db.init_db()
    return True


_boot()
ss = st.session_state

# =====================================================================
# Small helpers
# =====================================================================
def esc(x):
    return html.escape(str(x if x is not None else ""))


def wide(fn, *args, **kwargs):
    """Full-width widgets across Streamlit versions (width='stretch' is new, use_container_width is old)."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}):
        try:
            return fn(*args, **kwargs, **extra)
        except Exception:
            pass
    return fn(*args, **kwargs)


def _idx(options, val):
    return options.index(val) if val in options else 0


def flash(msg):
    ss.flash = msg


# =====================================================================
# UI strings (interface language) - clinical content is translated separately by the engine
# =====================================================================
LANG_MAP = {"EN": "en", "HI": "hi", "Hinglish": "en", "Tamil": "ta", "Telugu": "te", "Bengali": "bn"}
TRANSLATIONS = {
    "Welcome to CareFlow AI": {"en": "Welcome to CareFlow AI", "hi": "केयरफ्लो AI में आपका स्वागत है", "ta": "கேர்ஃப்ளோ AI க்கு வரவேற்கிறோம்", "te": "కేర్‌ఫ్లో AI కి స్వాగతం", "bn": "কেয়ারফ্লো AI-তে আপনাকে স্বাগত"},
    "Patient Name": {"en": "Patient Name", "hi": "मरीज का नाम", "ta": "நோயாளி பெயர்", "te": "రోగి పేరు", "bn": "রোগীর নাম"},
    "Age": {"en": "Age", "hi": "उम्र", "ta": "வயது", "te": "వయస్సు", "bn": "বয়স"},
    "Sex": {"en": "Sex", "hi": "लिंग", "ta": "பாலினம்", "te": "లింగం", "bn": "লিঙ্গ"},
    "Language": {"en": "Language", "hi": "भाषा", "ta": "மொழி", "te": "భాష", "bn": "ভাষা"},
    "Chief Complaint": {"en": "Chief Complaint", "hi": "मुख्य शिकायत", "ta": "முக்கிய புகார்", "te": "ప్రధాన సమస్య", "bn": "প্রধান অভিযোগ"},
    "Symptom Summary": {"en": "Symptom Summary", "hi": "लक्षण सारांश", "ta": "அறிகுறி சுருக்கம்", "te": "లక్షణాల సారాంశం", "bn": "লক্ষণ সারাংশ"},
    "Doctor's Note": {"en": "Doctor's message", "hi": "डॉक्टर का संदेश", "ta": "மருத்துவரின் செய்தி", "te": "డాక్టర్ సందేశం", "bn": "ডাক্তারের বার্তা"},
    "Prescribed Medicines": {"en": "Prescribed Medicines", "hi": "निर्धारित दवाएं", "ta": "பரிந்துரைக்கப்பட்ட மருந்துகள்", "te": "సూచించిన మందులు", "bn": "নির্ধারিত ওষুধপত্র"},
    "Your Diet Chart": {"en": "Your Diet Chart", "hi": "आपका डाइट चार्ट", "ta": "உங்கள் உணவு விளக்கப்படம்", "te": "మీ డైట్ చార్ట్", "bn": "আপনার ডায়েট চার্ট"},
}


def t(key):
    return TRANSLATIONS.get(key, {}).get(LANG_MAP.get(ss.get("ui_lang", "EN"), "en"), key)


# =====================================================================
# Session defaults + engine configuration
# =====================================================================
for _k, _v in {"role": None, "username": None, "step": 1, "intake": {}, "draft": {}, "voice": None, "ocr": None,
               "ui_lang": "EN", "preview": None, "submitted": None, "viewed": set(), "flash": None}.items():
    ss.setdefault(_k, _v)

if "cfg" not in ss:
    _cfg = dict(eng.DEFAULT_CFG)
    _models = eng.ollama_models(_cfg["host"])
    ss.ollama_models = _models
    if _models:
        _cfg["use_slm"] = True
        _pref = [m for m in _models if m.split(":")[0].startswith(("qwen2.5", "llama3.1", "gemma", "phi", "mistral"))]
        _cfg["model"] = (_pref or _models)[0]
    ss.cfg = _cfg


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_translate(text, lang, host, model, use_slm, allow_online):
    return eng.translate_text(text, lang, {"host": host, "model": model, "use_slm": use_slm, "allow_online": allow_online})


def tr(text, lang):
    c = ss.cfg
    return _cached_translate(text, lang, c["host"], c["model"], bool(c["use_slm"]), bool(c["allow_online"]))


# =====================================================================
# Design system
# =====================================================================
st.markdown("""
<style>
.stApp { background-color:#ffffff; color:#1a202c; font-family:'Inter',sans-serif; }
.header-banner { background:#e6f0ff; border-bottom:3px solid #2b6cb0; padding:18px 22px; border-radius:16px; margin-bottom:16px; }
.badge-hipaa { background:#2b6cb0; color:#fff; padding:6px 12px; border-radius:20px; font-weight:bold; font-size:12px; }
.badge-urgent { background:#e53e3e; color:#fff; padding:4px 10px; border-radius:6px; font-weight:bold; }
.badge-review { background:#dd6b20; color:#fff; padding:4px 10px; border-radius:6px; font-weight:bold; }
.badge-routine { background:#38a169; color:#fff; padding:4px 10px; border-radius:6px; font-weight:bold; }
.flag-red { border-left:4px solid #e53e3e; background:#fff5f5; padding:10px 12px; border-radius:4px; margin-bottom:8px; color:#9b2c2c; }
.flag-orange { border-left:4px solid #dd6b20; background:#fffaf0; padding:10px 12px; border-radius:4px; margin-bottom:8px; color:#9c4221; }
.flag-yellow { border-left:4px solid #d69e2e; background:#fffff0; padding:10px 12px; border-radius:4px; margin-bottom:8px; color:#744210; }
.flag-blue { border-left:4px solid #3182ce; background:#ebf8ff; padding:10px 12px; border-radius:4px; margin-bottom:8px; color:#2c5282; }
.src { font-size:12px; opacity:.85; }
.srcbox { background:#f7fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px; margin-bottom:10px; font-size:14px; color:#1a202c; }
.srcbox mark { background:#fefcbf; padding:0 2px; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

h1, h2 = st.columns([4, 1])
with h2:  # rendered first in code so t() sees the new language on the same run
    st.selectbox("🌐 Language / भाषा", ["EN", "HI", "Hinglish", "Tamil", "Telugu", "Bengali"], key="ui_lang")
with h1:
    st.markdown(
        '<div class="header-banner"><div style="display:flex;justify-content:space-between;align-items:center;">'
        '<h2 style="margin:0;color:#2b6cb0;font-weight:800;">%s</h2>'
        '<span class="badge-hipaa">🔒 Privacy-by-design demo</span></div>'
        '<p style="margin:5px 0 0 0;font-size:13px;color:#4a5568;">Human-in-the-loop AI triage assistant - it never diagnoses; a clinician decides.</p></div>'
        % esc(t("Welcome to CareFlow AI")), unsafe_allow_html=True)

if ss.flash:
    st.success(ss.flash)
    ss.flash = None


# =====================================================================
# Shared rendering helpers
# =====================================================================
def urgency_badge(level):
    cls = {"URGENT": "badge-urgent", "REVIEW SOON": "badge-review"}.get(level, "badge-routine")
    txt = {"URGENT": "🚨 URGENT - PRIORITY 1", "REVIEW SOON": "🟠 REVIEW SOON - PRIORITY 2"}.get(level, "🟢 ROUTINE - PRIORITY 3")
    return '<span class="%s">%s</span>' % (cls, txt)


def ptext(label, value):
    """Render patient-derived text safely (escaped) with a bold label."""
    st.markdown('<p style="margin:2px 0"><b>%s</b> %s</p>' % (esc(label), esc(value)), unsafe_allow_html=True)


FLAG_CLASS = {"urgent": "flag-red", "security": "flag-red", "conflict": "flag-orange", "review": "flag-yellow",
              "uncertain": "flag-yellow", "missing": "flag-yellow", "info": "flag-blue"}
FLAG_ICON = {"urgent": "🚨", "security": "🛡️", "conflict": "🟠", "review": "⚠️", "uncertain": "❔", "missing": "📭", "info": "ℹ️"}
FLAG_ORDER = {"urgent": 0, "security": 1, "conflict": 2, "review": 3, "uncertain": 4, "missing": 5, "info": 6}


def render_flags(case):
    ai, sources = case["ai_summary"], case["sources"]
    if not ai["flags"]:
        st.caption("No flags raised.")
        return
    for f in sorted(ai["flags"], key=lambda x: FLAG_ORDER.get(x["kind"], 9)):
        src = next((s for s in sources if s["id"] == f.get("source_id")), None)
        line = ""
        if src:
            q = " - “%s”" % esc(f["quote"]) if f.get("quote") else ""
            line = '<br><span class="src">Source %s · %s%s</span>' % (esc(src["id"]), esc(src["label"]), q)
        st.markdown('<div class="%s"><b>%s %s</b><br>%s%s</div>' % (
            FLAG_CLASS.get(f["kind"], "flag-yellow"), FLAG_ICON.get(f["kind"], ""), esc(f["title"]), esc(f["detail"]), line),
            unsafe_allow_html=True)


def render_sources(case, key):
    if not st.checkbox("🔗 Show original sources with highlighted evidence", key="src_%s" % key):
        return
    ai = case["ai_summary"]
    for s in case["sources"]:
        spans = [f["span"] for f in ai["flags"] if f.get("source_id") == s["id"] and f.get("span")]
        spans += [f["span"] for f in ai["findings"] if f.get("source_id") == s["id"] and f.get("span")]
        meta = []
        if s.get("confidence") is not None:
            meta.append("confidence %d%%" % int(s["confidence"] * 100))
        if s.get("engine"):
            meta.append(str(s["engine"]))
        st.markdown("**%s · %s**" % (s["id"], s["label"]) + ("  \n*%s*" % ", ".join(meta) if meta else ""))
        st.markdown('<div class="srcbox">%s</div>' % eng.highlight_html(s["text"], spans), unsafe_allow_html=True)
    qa = case.get("patient_input", {}).get("followup_qa") or []
    if qa:
        st.markdown("**Follow-up questions asked → answers**")
        for item in qa:
            ptext("Q%d: %s" % (item["n"], item["q"]), "→ " + item["a"])


def render_ai_panel(case):
    ai = case["ai_summary"]
    ptext("Extracted chief complaint:", ai["chief_complaint"])
    st.markdown(urgency_badge(ai["urgency"]), unsafe_allow_html=True)
    if ai["urgency_reasons"]:
        st.caption("Why: " + "; ".join(ai["urgency_reasons"]))
    c1, c2 = st.columns(2)
    c1.metric("Extraction confidence", "%d%%" % ai["confidence"], help="How complete and consistent the information is - NOT diagnostic confidence. " + "; ".join(ai["confidence_notes"]))
    c2.metric("Analysis mode", ai["mode"], help="Rules always set the safety floor. The local model can only raise urgency.")
    if ai["slm_urgency"] and ai["slm_urgency"] != ai["rule_urgency"]:
        st.caption("Rules: %s · AI model: %s" % (ai["rule_urgency"], ai["slm_urgency"]))
    ptext("Summary (English):", ai["english_summary"])
    tl = ai["timeline"]
    ptext("Timeline:", "duration %s · progression %s" % (tl["duration"] or "not stated", ", ".join(tl["progression"]) or "not stated"))
    n2 = ai["news2"]
    if n2["applicable"] and n2["parts"]:
        st.caption("Vitals early-warning score (NEWS2-style): **%s** · %s · %s" % (
            n2["total"], ", ".join("%s %d" % (k, v) for k, v in n2["parts"].items()), n2["note"]))
    elif not n2["applicable"]:
        st.caption(n2["note"])
    if ai["findings"]:
        rows = [{"Finding": f["label"], "Status": "denied" if f["negated"] else "reported", "Source": f["source_id"] or "-",
                 "Evidence": f.get("span") or f.get("quote") or "", "Found by": f["origin"].upper() + ("" if f.get("grounded", True) else " (unverified)")}
                for f in ai["findings"]]
        wide(st.dataframe, pd.DataFrame(rows), hide_index=True)
    st.markdown("**Flags (with source links)**")
    render_flags(case)
    if ai["followups"]:
        st.markdown("**AI-suggested follow-up questions**")
        for q in ai["followups"]:
            st.markdown("- " + q["en"])


def render_case_overview(case, key):
    pi, ai = case["patient_input"], case["ai_summary"]
    ptext("Patient:", "%s %s · %s %s · %s · prefers %s" % (pi.get("title", ""), pi.get("name", ""), pi.get("age"), pi.get("age_unit"), pi.get("sex"), pi.get("lang")))
    st.markdown(urgency_badge(ai["urgency"]), unsafe_allow_html=True)
    ptext("Chief complaint (AI):", ai["chief_complaint"])
    ptext("History:", pi.get("history") or "None stated")
    for f in sorted(ai["flags"], key=lambda x: FLAG_ORDER.get(x["kind"], 9))[:6]:
        st.markdown('<div class="%s"><b>%s %s</b> - %s</div>' % (FLAG_CLASS.get(f["kind"], "flag-yellow"), FLAG_ICON.get(f["kind"], ""), esc(f["title"]), esc(f["detail"])), unsafe_allow_html=True)
    if ai["followups"]:
        st.markdown("**Questions to ask the patient:** " + " · ".join(q["en"] for q in ai["followups"][:4]))
    render_sources(case, key)


def log_view(cid):
    tag = (ss.username, cid)
    if tag not in ss.viewed:
        ss.viewed.add(tag)
        db.audit(ss.username, ss.role, "case_viewed", cid)


def vitals_inputs(prefix, defaults):
    cols = st.columns(5)
    raw = {}
    spec = [("temp", "Temp (°C)", "36.8"), ("bp", "BP (mmHg)", "120/80"), ("spo2", "SpO2 (%)", "98"), ("hr", "Heart rate (bpm)", "72"), ("rr", "Resp. rate (/min)", "16")]
    for col, (k, label, ph) in zip(cols, spec):
        raw[k] = col.text_input(label, value=defaults.get(k, ""), placeholder=ph, key="%s_%s" % (prefix, k))
    return raw


def data_of(case):
    return db.data_of(case)


# =====================================================================
# Sidebar (all roles)
# =====================================================================
def sidebar_common():
    st.sidebar.markdown("👤 **%s**  \n%s" % (ss.username, ss.role))
    if st.sidebar.button("🔒 LOG OUT"):
        db.audit(ss.username, ss.role, "logout")
        ss.role, ss.username, ss.step, ss.preview, ss.voice, ss.ocr = None, None, 1, None, None, None
        st.rerun()
    with st.sidebar.expander("⚙️ AI engine settings"):
        cfg = ss.cfg
        ss.setdefault("cfg_host", cfg["host"])
        cfg["host"] = st.text_input("Ollama host", key="cfg_host")
        if st.button("↻ Detect local models"):
            ss.ollama_models = eng.ollama_models(cfg["host"])
        models = ss.get("ollama_models", [])
        if models:
            if ss.get("cfg_model") not in models:
                ss["cfg_model"] = cfg["model"] if cfg["model"] in models else models[0]
            cfg["model"] = st.selectbox("Model (<20B params recommended)", models, key="cfg_model")
        else:
            ss.setdefault("cfg_model", cfg["model"])
            cfg["model"] = st.text_input("Model name", key="cfg_model")
            st.caption("No local Ollama detected → rules-only mode. To enable the SLM: install Ollama, run `ollama pull qwen2.5:7b-instruct`, click Detect.")
        ss.setdefault("cfg_use_slm", cfg["use_slm"])
        cfg["use_slm"] = st.checkbox("Use local SLM for extraction + translation", key="cfg_use_slm")
        ss.setdefault("cfg_online", cfg["allow_online"])
        cfg["allow_online"] = st.checkbox("Allow ONLINE translation (text leaves this device)", key="cfg_online")
    if ss.cfg["use_slm"] and ss.get("ollama_models"):
        st.sidebar.caption("🧠 Rules + local SLM (%s)" % ss.cfg["model"])
    else:
        st.sidebar.caption("⚙️ Rules-only mode")
    st.sidebar.caption("🔒 Identifiers are auto-redacted. Only extracted text is stored - uploaded files and audio are discarded.")


# =====================================================================
# LOGIN
# =====================================================================
PATIENTS = ["Aarav Sharma", "Priya Patel", "Ramesh Kumar", "Kabir Singh", "Anaya Verma"]


def login_screen():
    with st.container(border=True):
        st.markdown("### 🔐 Central Login Gateway")
        role = st.radio("Select portal:", ["Patient View", "Emergency Nurse Node", "Attending Doctor (MD) View"], horizontal=True)
        if role == "Patient View":
            user = st.selectbox("Profile (demo patients):", PATIENTS)
        elif role == "Emergency Nurse Node":
            user = st.selectbox("Profile:", ["Nurse Sunita"])
        else:
            user = st.selectbox("Profile:", ["Dr. Sharma - Cardiologist", "Dr. Gupta - General Physician", "Dr. Meena - Dietician"])
        pw = st.text_input("Security access code:", type="password", placeholder="Demo code: 1234")
        st.caption("Demo authentication only. A real deployment needs SSO/OTP and per-user credentials.")
        if wide(st.button, "🔐 LOG IN", type="primary"):
            if pw == os.environ.get("CAREFLOW_DEMO_PASSWORD", "1234"):
                ss.role, ss.username = role, user
                db.audit(user, role, "login")
                st.rerun()
            else:
                st.error("Access refused: invalid credentials.")


# =====================================================================
# PATIENT
# =====================================================================
LANGS = ["English", "Hindi", "Hinglish", "Tamil", "Telugu", "Bengali"]
BODY_AREAS = ["Chest Pain", "Abdominal Pain", "Migraine / Headache", "Lower Back Pain", "Throat", "Arms / Legs"]
HIST_OPTS = ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Tuberculosis", "Others"]


def emergency_notice(case_like):
    keys = {f["key"] for f in case_like["findings"] if not f["negated"]}
    if "self_harm" in keys:
        st.error("💙 You are not alone. Please reach out now: **Tele-MANAS 14416** (free, India) or call **112**. A member of our team has been alerted.")
    if case_like["urgency"] == "URGENT":
        st.error("🚨 Your symptoms may need immediate attention. **Tell the nearest staff member right now**, or call **112** (emergency) / **108** (ambulance). Do not wait for the screen to update.")


def patient_step1():
    it = ss.intake
    with st.container(border=True):
        st.markdown("##### Step 1: Patient information")
        c1, c2, c3 = st.columns([1, 2, 1])
        titles = ["Mr.", "Ms.", "Mrs."]
        title = c1.selectbox("%s title*" % t("Patient Name"), titles, index=_idx(titles, it.get("title")))
        c2.text_input(t("Patient Name"), value=ss.username, disabled=True)
        a1, a2 = c3.columns(2)
        age = a1.number_input("%s*" % t("Age"), min_value=0, max_value=120, value=int(it.get("age", 0)))
        units = ["Years", "Months", "Days"]
        unit = a2.selectbox("Unit*", units, index=_idx(units, it.get("age_unit")))
        s1, s2 = st.columns(2)
        sexes = ["Male", "Female", "Other"]
        sex = s1.radio("%s*" % t("Sex"), sexes, index=_idx(sexes, it.get("sex")), horizontal=True)
        lang = s2.selectbox("%s*" % t("Language"), LANGS, index=_idx(LANGS, it.get("lang")))
        consent = st.checkbox("I consent to AI-assisted triage. My data is used only for my care, identifiers are auto-redacted, and I can withdraw consent and delete my data at any time (tab: My reports).*",
                              value=bool(it.get("consent_given")))
    with st.expander("+ Add medical context (optional) - helps the AI prioritise you better"):
        hist = st.multiselect("Past medical history:", HIST_OPTS, default=[h for h in it.get("history_list", []) if h in HIST_OPTS])
        allergies = st.text_input("Drug allergies (if any):", value=it.get("allergies", ""))
        meds = st.text_input("Regular medicines (if any):", value=it.get("medications", ""))
        st.caption("Only enter readings you actually measured - leave the rest blank.")
        raw = vitals_inputs("p1", {"temp": it.get("v_temp", ""), "bp": it.get("v_bp", ""), "spo2": it.get("v_spo2", ""), "hr": it.get("v_hr", ""), "rr": it.get("v_rr", "")})
        for msg in eng.parse_vitals(raw)[1]:
            st.warning(msg)
    if wide(st.button, "Continue to symptom assessment ➔", type="primary"):
        if not consent:
            st.error("Consent is required to continue.")
        elif age <= 0:
            st.error("Please enter a valid age.")
        else:
            ss.intake = {"title": title, "name": ss.username, "age": int(age), "age_unit": unit, "sex": sex, "lang": lang,
                         "history": ", ".join(hist) if hist else "None", "history_list": hist, "allergies": allergies.strip(),
                         "medications": meds.strip(), "v_temp": raw["temp"].strip(), "v_bp": raw["bp"].strip(), "v_spo2": raw["spo2"].strip(),
                         "v_hr": raw["hr"].strip(), "v_rr": raw["rr"].strip(), "vitals_loaded": any(v.strip() for v in raw.values()),
                         "consent_given": True}
            ss.step = 2
            st.rerun()


def patient_step2():
    d = ss.draft
    with st.container(border=True):
        st.markdown("##### Step 2: Tell us your symptoms - type, speak or upload a report")
        tab_t, tab_v, tab_r = st.tabs(["⌨️ Type symptoms", "🎙️ Record voice", "📄 Upload report"])
        with tab_t:
            typed = st.text_area("Describe your symptoms in your own language (e.g. 'Mujhe 3 din se bukhar hai'):", value=d.get("typed", ""), height=110)
            onset_text = st.text_input("When did it start? (optional)", value=d.get("onset_text", ""), placeholder="e.g. 3 days ago")
            opts = ["Not sure", "Getting worse", "About the same", "Getting better"]
            change = st.radio("How has it changed since it started?", opts, index=_idx(opts, d.get("change")), horizontal=True)
            sev_on = st.checkbox("I want to rate how severe it feels", value=bool(d.get("severity")))
            severity = st.slider("Severity (1 = very mild, 10 = worst)", 1, 10, int(d.get("severity") or 5)) if sev_on else None
        with tab_v:
            asr_ok = eng.asr_available()
            st.caption("Speech-to-text runs locally on this machine." if asr_ok else
                       "Local speech-to-text (faster-whisper) is not installed: `pip install faster-whisper`. You can still type the transcript below.")
            audio = st.audio_input("Record a voice note") if hasattr(st, "audio_input") else None
            up_a = st.file_uploader("...or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "webm"], key="audio_up")
            blob, fname = (audio.getvalue(), "voice.wav") if audio else ((up_a.getvalue(), up_a.name) if up_a else (None, None))
            if blob and asr_ok and st.button("📝 Transcribe locally"):
                hint = {"Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Bengali": "bn"}.get(ss.intake.get("lang"))
                with st.spinner("Transcribing locally..."):
                    res = eng.transcribe_audio(blob, suffix=os.path.splitext(fname)[1] or ".wav", language=hint)
                if res["error"]:
                    st.error(res["error"])
                else:
                    res["orig_text"] = res["text"]
                    ss.voice = res
                    ss["voice_text"] = res["text"]
            v = ss.voice
            if v:
                if "voice_text" not in ss:
                    ss["voice_text"] = v.get("text", "")
                for sg in [x for x in v.get("segments", []) if x["conf"] < 0.5][:3]:
                    st.warning("Unsure about the speech at %ss (%d%% confidence): “%s”. Please check and correct it below." % (sg["start"], int(sg["conf"] * 100), sg["text"]))
                if v.get("language_probability") is not None and v["language_probability"] < 0.7:
                    st.warning("Unsure which language was spoken (%s, %d%%)." % (v.get("language"), int(v["language_probability"] * 100)))
            voice_text = st.text_area("Transcript (check it and correct any mistakes):", key="voice_text", height=90) if (v or blob) else ""
        with tab_r:
            ocr_ok = eng.ocr_available()
            st.caption("Reports are read locally with OCR; the file itself is discarded - only the text is kept." if ocr_ok else
                       "Local OCR (Tesseract) was not detected. You can paste the report text below, or use the sample report to try the flow.")
            up_f = st.file_uploader("Drop medical report / prescription / lab image here:", type=["png", "jpg", "jpeg", "pdf"], key="report_up")
            if up_f is not None:
                sig = "%s:%s" % (up_f.name, up_f.size)
                if ss.get("ocr_sig") != sig:
                    with st.spinner("Reading the report locally..."):
                        res = eng.ocr_file(up_f.getvalue(), up_f.name)
                    res["orig_text"] = res["text"]
                    ss.ocr_sig, ss.ocr, ss["ocr_text"] = sig, res, res["text"]
            if st.button("Use a sample lab report (simulated text, for demo)"):
                txt = eng.DEMO_REPORT_TEMPLATE.format(name=ss.username)
                ss.ocr = {"text": txt, "orig_text": txt, "confidence": 0.82, "filename": "sample_cbc_report (demo)", "engine": "simulated",
                          "simulated": True, "low_conf_words": [], "error": None}
                ss["ocr_text"] = txt
            o = ss.ocr
            if o:
                if "ocr_text" not in ss:
                    ss["ocr_text"] = o.get("text", "")
                if o.get("error"):
                    st.error(o["error"])
                elif o.get("confidence") is not None and o["confidence"] < 0.6:
                    st.warning("This report is hard to read (OCR confidence %d%%). Please check the text below." % int(o["confidence"] * 100))
                else:
                    st.success("✓ Report text loaded. Please check it below.")
            ocr_text = st.text_area("Report text (edit if the reading is wrong, or paste it):", key="ocr_text", height=120) if (o or not ocr_ok) else ""
        body = st.multiselect("Where does it hurt? (optional)", BODY_AREAS, default=[b for b in d.get("body_areas", []) if b in BODY_AREAS])
    b1, b2 = st.columns(2)
    if b1.button("⬅ Back to step 1"):
        ss.step = 1
        st.rerun()
    if wide(b2.button, "Review what we understood ➔", type="primary"):
        if not (typed.strip() or voice_text.strip() or ocr_text.strip()):
            st.error("Please describe your symptoms (type, speak, or upload a report).")
            return
        voice = None
        if voice_text.strip():
            v = ss.voice or {}
            edited = bool(v) and voice_text.strip() != v.get("orig_text", "").strip()
            voice = {"text": voice_text, "segments": [] if (edited or not v) else v.get("segments", []),
                     "mean_conf": None if (edited or not v) else v.get("mean_conf"), "language": v.get("language"),
                     "language_probability": v.get("language_probability")}
        ocr = None
        if ocr_text.strip():
            o = ss.ocr or {}
            edited = bool(o) and ocr_text.strip() != o.get("orig_text", "").strip()
            ocr = {"text": ocr_text, "confidence": None if (edited or not o) else o.get("confidence"),
                   "filename": o.get("filename") or "pasted report text", "engine": o.get("engine") or "manual entry",
                   "low_conf_words": [] if edited else o.get("low_conf_words", []), "simulated": bool(o.get("simulated")) and not edited}
        ss.draft = {"typed": typed, "onset_text": onset_text, "change": change, "severity": severity, "body_areas": body, "voice": voice, "ocr": ocr}
        ss.step = 3
        st.rerun()


def patient_step3():
    it, d = ss.intake, ss.draft
    sig = hashlib.sha256(json.dumps([it, d], sort_keys=True, default=str).encode()).hexdigest()
    pv = ss.preview
    if not pv or pv["sig"] != sig:
        with st.spinner("Structuring your information (locally)..."):
            intake = {**it, "onset_text": d.get("onset_text", ""), "change": d.get("change"), "severity": d.get("severity"), "body_areas": d.get("body_areas", [])}
            sources, red = eng.build_sources(intake, typed=d.get("typed", ""), voice=d.get("voice"), ocr=d.get("ocr"))
            ai = eng.run_analysis(intake, sources, ss.cfg)
        pv = ss.preview = {"sig": sig, "intake": intake, "sources": sources, "ai": ai, "pii": red}
    ai, lang = pv["ai"], it.get("lang", "English")
    with st.container(border=True):
        st.markdown("##### Step 3: Check what we understood, then answer a few questions")
        emergency_notice(ai)
        summ, mode = tr(ai["english_summary"], lang)
        ptext("%s:" % t("Symptom Summary"), summ)
        if mode not in ("original",):
            st.caption("Translation: %s - machine translated, may be imperfect." % mode)
        if lang in ("Hindi", "Tamil", "Telugu", "Bengali") and mode == "unavailable":
            st.caption("Translation is unavailable right now (enable the local model or online translation in the sidebar). Showing English.")
        reported = [f["label"] for f in ai["findings"] if not f["negated"]]
        if reported:
            st.markdown("**We noted:** " + ", ".join(dict.fromkeys(reported)))
        if pv["pii"]:
            st.info("🔒 %d personal identifier(s) (phone / email / ID number) were removed from your text for privacy." % pv["pii"])
        st.caption("If anything above is wrong, go back and edit it. Your answers help the care team prioritise you.")
        for q in ai["followups"]:
            if q["origin"] == "rule":
                qt = eng.followup_text(q, lang)
                if lang in ("Tamil", "Telugu", "Bengali"):
                    qt = tr(q["en"], lang)[0]
            else:
                qt = tr(q["en"], lang)[0] if lang in ("Hindi", "Tamil", "Telugu", "Bengali") else q["en"]
            st.text_area(qt, key="fu_%s" % q["id"], height=68)
    b1, b2 = st.columns(2)
    if b1.button("⬅ Back to step 2"):
        ss.step = 2
        st.rerun()
    if wide(b2.button, "✅ Submit to care team", type="primary"):
        qa, lines = [], []
        for q in ai["followups"]:
            a = (ss.get("fu_%s" % q["id"]) or "").strip()
            if a:
                n = len(qa) + 1
                qa.append({"n": n, "q": q["en"], "a": a})
                lines.append("Follow-up %d: %s" % (n, a))   # only the patient's words go into the evidence text
        intake = {**pv["intake"], "symptom_notes": d.get("typed", ""), "followup_qa": qa}
        sources, red = eng.build_sources(intake, typed=d.get("typed", ""), voice=d.get("voice"), ocr=d.get("ocr"), followup="\n".join(lines))
        for item in qa:
            item["a"], _ = eng.redact_pii(item["a"])
        ai2 = eng.run_analysis(intake, sources, ss.cfg, prior=ai)
        data = {"patient_input": intake, "sources": sources, "ai_summary": ai2, "nurse_output": {}, "doctor_output": {},
                "consent": {"granted": True, "policy_version": db.POLICY_VERSION, "ts": eng.now_iso()}}
        cid = db.create_case(ss.username, data, ai2["urgency"], ai2["news2"]["total"])
        db.add_consent(ss.username, cid)
        db.audit(ss.username, "Patient", "case_submitted", cid, "pii_redacted=%d" % red)
        db.audit("CareFlow-AI", "AI", "ai_analysis", cid, "%s; urgency=%s; confidence=%d" % (ai2["mode"], ai2["urgency"], ai2["confidence"]))
        ss.submitted = {"case_id": cid, "ai": {"findings": ai2["findings"], "urgency": ai2["urgency"]}}
        ss.step, ss.preview, ss.draft, ss.voice, ss.ocr = 4, None, {}, None, None
        for k in ("voice_text", "ocr_text", "ocr_sig"):
            ss.pop(k, None)
        st.rerun()


def patient_step4():
    sub = ss.submitted or {}
    with st.container(border=True):
        st.success("🎉 Your information has reached the care team. Case ID: **%s**" % sub.get("case_id", ""))
        if sub.get("ai"):
            emergency_notice(sub["ai"])
        st.caption("A nurse will verify your details and a doctor will review your case. Check the 'My reports' tab for updates.")
    if st.button("Start another record"):
        ss.step, ss.intake, ss.submitted = 1, {}, None
        st.rerun()


def patient_reports():
    cases = db.list_cases(patient_user=ss.username)
    st.markdown("<h4>📋 My reports - %s &amp; prescriptions</h4>" % esc(t("Your Diet Chart")), unsafe_allow_html=True)
    if not cases:
        st.info("No records yet.")
    for c in cases:
        with st.expander("%s - %s - submitted %s UTC" % (c["case_id"], c["status"], c["created_ts"]), expanded=(c["status"] == "Reviewed")):
            if c["status"] != "Reviewed":
                st.info("ℹ️ Your case is with the care team (status: %s). Nothing has been published yet." % c["status"])
                continue
            do, lang = c["doctor_output"], c["patient_input"].get("lang", "English")
            msg = do.get("patient_message", "")
            ptext("%s:" % t("Doctor's Note"), msg)
            if lang in ("Hindi", "Tamil", "Telugu", "Bengali") and msg:
                tmsg, mode = tr(msg, lang)
                if mode != "unavailable":
                    ptext("(%s)" % lang, tmsg)
                    st.caption("Machine-translated (%s). If unsure, ask the staff." % mode)
            st.markdown("**Follow-up:** %s" % esc(do.get("follow_up", "Not required")), unsafe_allow_html=True)
            st.markdown("**💊 %s**" % t("Prescribed Medicines"))
            if do.get("prescription"):
                st.table(pd.DataFrame(do["prescription"]))
            else:
                st.caption("No medicines were prescribed.")
            st.markdown("**🍱 %s**" % t("Your Diet Chart"))
            if any(do.get("diet_chart", {}).values()):
                st.table(pd.DataFrame([{"Meal / note": k, "Plan": v} for k, v in do["diet_chart"].items() if v]))
            else:
                st.caption("No diet chart added.")
    with st.container(border=True):
        st.markdown("##### 🔒 Your data and consent")
        st.caption("We store the text of your answers (identifiers redacted) and the analysis. Uploaded files and audio are never stored. You can withdraw consent and erase your clinical data at any time; only an empty case-ID tombstone and the audit trail (no medical content) remain.")
        sure = st.checkbox("I understand this cannot be undone.", key="del_ok")
        if st.button("🗑️ Withdraw consent and delete my data"):
            if not sure:
                st.error("Please tick the confirmation box first.")
            else:
                n = db.delete_patient_data(ss.username, ss.username, "Patient")
                flash("Consent withdrawn. %d case record(s) erased." % n)
                st.rerun()


def patient_view():
    sidebar_common()
    open_cases = [c for c in db.list_cases(patient_user=ss.username) if c["status"] != "Reviewed"]
    tab1, tab2 = st.tabs(["📥 Create new triage record", "📋 My reports"])
    with tab1:
        if open_cases and ss.step < 4:
            st.warning("You already have an active case (%s, status: %s). You can still add a new record." % (open_cases[0]["case_id"], open_cases[0]["status"]))
        st.markdown("<h4>Progress: step %d / 4</h4>" % min(ss.step, 4), unsafe_allow_html=True)
        {1: patient_step1, 2: patient_step2, 3: patient_step3, 4: patient_step4}[ss.step]()
    with tab2:
        patient_reports()


# =====================================================================
# NURSE
# =====================================================================
def nurse_view():
    sidebar_common()
    st.markdown("### 📋 Emergency Department - triage verification queue")
    cases = db.list_cases(statuses=["Submitted"])
    if not cases:
        st.success("✓ No cases are waiting for nurse verification.")
        return
    st.caption("%d case(s) waiting - sorted by priority (urgent first)." % len(cases))
    for c in cases:
        cid, ai = c["case_id"], c["ai_summary"]
        icon = {"URGENT": "🔴", "REVIEW SOON": "🟠"}.get(ai["urgency"], "🟢")
        with st.expander("%s %s - %s - %s" % (icon, cid, c["patient_name"], ai["urgency"]), expanded=(ai["urgency"] == "URGENT")):
            log_view(cid)
            render_case_overview(c, "n_" + cid)
            st.markdown("**Your verification**")
            notes = st.text_area("Verification notes (required):", key="n_note_%s" % cid, height=90)
            st.caption("Re-measure and enter vitals. Leave blank what you did not measure - nothing is pre-filled.")
            raw = vitals_inputs("n_%s" % cid, {})
            for msg in eng.parse_vitals(raw)[1]:
                st.warning(msg)
            raise_u = st.checkbox("Raise to URGENT (clinical concern)", key="n_raise_%s" % cid)
            if wide(st.button, "⚡ ESCALATE TO DOCTOR QUEUE", key="n_esc_%s" % cid, type="primary"):
                if not notes.strip():
                    st.error("Please add verification notes before escalating.")
                    continue
                sources = eng.add_nurse_source(c["sources"], raw, notes)
                ai2 = eng.run_analysis(c["patient_input"], sources, ss.cfg, nurse={"vitals": raw, "raise_urgent": raise_u}, prior=ai)
                data = data_of(c)
                data.update(sources=sources, ai_summary=ai2,
                            nurse_output={"triage_notes": notes.strip(), "vitals_recheck": {k: v for k, v in raw.items() if v.strip()},
                                          "nurse": ss.username, "ts": eng.now_iso(), "raised_urgent": raise_u})
                try:
                    db.update_case(cid, data, "Pending Doctor Review", ai2["urgency"], ai2["news2"]["total"], expected_version=c["version"])
                except db.ConflictError as e:
                    st.error(str(e))
                    continue
                db.audit(ss.username, ss.role, "escalated_to_doctor", cid, "urgency=%s; raised=%s" % (ai2["urgency"], raise_u))
                flash("Case %s escalated to the doctor queue." % cid)
                st.rerun()


# =====================================================================
# DOCTOR
# =====================================================================
DIET = {
    "None": ("", "", "", ""),
    "Diabetic Diet Profile": ("Oats Upma + Sugar-free Green Tea", "Brown Rice + Dal Tadka + Roasted Salad + Curd", "Sprouted Moong Chaat", "Sugar drinks, White bread, Sweets"),
    "Hypertension Control Diet": ("Low-salt Ragi Idli + Almonds", "Whole Wheat Roti + Boiled Palak Dal + Cucumber Raita", "Roasted Makhana (no added salt)", "High-sodium foods, Pickles, Papad"),
    "Fever recovery (light diet)": ("Soft moong dal khichdi + warm water", "Rice + thin dal + boiled vegetables + curd", "Coconut water / ORS + banana or papaya", "Oily or spicy food, Street food, Cold drinks"),
}
FOLLOW_UPS = ["Not required", "In 3 days", "In 7 days", "In 14 days", "In 1 month"]


def _apply_tpl(cid):
    for k, v in zip("maev", DIET[ss["tpl_%s" % cid]]):
        ss["diet_%s_%s" % (k, cid)] = v


def _diet_inputs(cid):
    for k in "maev":
        ss.setdefault("diet_%s_%s" % (k, cid), "")
    st.selectbox("Pre-loaded dietary template:", list(DIET), key="tpl_%s" % cid, on_change=_apply_tpl, args=(cid,))
    out = {"Morning": st.text_input("Morning:", key="diet_m_%s" % cid), "Afternoon": st.text_input("Afternoon:", key="diet_a_%s" % cid),
           "Evening": st.text_input("Evening:", key="diet_e_%s" % cid), "AVOID": st.text_input("⚠️ AVOID / restrictions:", key="diet_v_%s" % cid)}
    return out


def doctor_actions(c):
    cid, ai = c["case_id"], c["ai_summary"]
    with st.container(border=True):
        st.markdown("##### ⚙️ Clinician review & decision (human-in-the-loop)")
        st.caption("The note below is an AI draft. Edit it as needed - nothing reaches the patient until you approve.")
        ss.setdefault("note_%s" % cid, ai["draft_note"])
        note = st.text_area("Clinical triage / referral note (internal, editable):", key="note_%s" % cid, height=280)
        if note.strip() != ai["draft_note"].strip():
            with st.expander("🔍 Your edits vs the AI draft"):
                diff = difflib.unified_diff(ai["draft_note"].splitlines(), note.splitlines(), fromfile="AI draft", tofile="Clinician final", lineterm="")
                st.code("\n".join(diff) or "(whitespace only)", language="diff")
        levels = list(eng.LEVELS)
        final_u = st.selectbox("Final triage level:", levels, index=levels.index(ai["urgency"]), key="fu_%s" % cid)
        reason = st.text_input("Reason for overriding the AI/rule level (required):", key="ov_%s" % cid) if final_u != ai["urgency"] else ""
        pmsg = st.text_area("Message to the patient (plain language - auto-translated for them):", key="pm_%s" % cid, height=90)
        st.markdown("**💊 Prescription**")
        meds_df = wide(st.data_editor, pd.DataFrame(columns=["Medicine Name", "Dosage", "Duration"]), num_rows="dynamic", key="meds_%s" % cid)
        st.markdown("**🍱 Indian dietary recovery chart**")
        diet = _diet_inputs(cid)
        fu = st.selectbox("Follow-up appointment:", FOLLOW_UPS, key="fup_%s" % cid)
        ack = st.checkbox("I have reviewed the AI summary, flags and original sources, and I take clinical responsibility for this note.", key="ack_%s" % cid)
        if wide(st.button, "🚀 APPROVE & RELEASE TO PATIENT", key="ap_%s" % cid, type="primary"):
            errs = []
            if not note.strip():
                errs.append("The clinical note is empty.")
            if not ack:
                errs.append("Please confirm that you reviewed the AI content.")
            if final_u != ai["urgency"] and not reason.strip():
                errs.append("Please give a reason for overriding the triage level.")
            if not pmsg.strip():
                errs.append("Please add a short message for the patient.")
            if errs:
                for e in errs:
                    st.error(e)
                return
            meds = [r for r in meds_df.fillna("").to_dict("records") if str(r.get("Medicine Name", "")).strip()]
            edited = note.strip() != ai["draft_note"].strip()
            data = data_of(c)
            data["doctor_output"] = {"referral_note": note.strip(), "ai_draft_note": ai["draft_note"], "edited_by_clinician": edited,
                                     "final_urgency": final_u, "ai_urgency": ai["urgency"], "override_reason": reason.strip(),
                                     "patient_message": pmsg.strip(), "prescription": meds, "diet_chart": diet, "follow_up": fu,
                                     "reviewer": ss.username, "reviewed_ts": eng.now_iso()}
            try:
                db.update_case(cid, data, "Reviewed", final_u, ai["news2"]["total"], expected_version=c["version"])
            except db.ConflictError as e:
                st.error(str(e))
                return
            db.audit(ss.username, ss.role, "case_approved", cid, "edited=%s; override=%s; medicines=%d" % (edited, final_u != ai["urgency"], len(meds)))
            flash("Case %s approved and released to the patient." % cid)
            st.rerun()


def doctor_view():
    sidebar_common()
    if "Dietician" in ss.username:
        return dietician_view()
    cases = db.list_cases()
    open_cases = [c for c in cases if c["status"] in ("Submitted", "Pending Doctor Review")]
    m1, m2, m3 = st.columns(3)
    m1.metric("Open cases", len(open_cases))
    m2.metric("🔴 Urgent (open)", sum(1 for c in open_cases if c["ai_summary"]["urgency"] == "URGENT"))
    m3.metric("⏳ Awaiting doctor", sum(1 for c in open_cases if c["status"] == "Pending Doctor Review"))
    show_rev = st.checkbox("Also show reviewed cases", value=False)
    queue = cases if show_rev else open_cases
    if not queue:
        st.success("✓ Queue is empty.")
        return
    st.markdown("<h4>📋 Prioritised triage queue (most urgent first)</h4>", unsafe_allow_html=True)
    rows = [{"Priority": i + 1, "Case": c["case_id"], "Patient": c["patient_name"], "Urgency": c["ai_summary"]["urgency"], "Status": c["status"],
             "NEWS2": c["news2"] if c["news2"] is not None else "n/a", "Alerts": sum(1 for f in c["ai_summary"]["flags"] if f["kind"] in ("urgent", "security", "conflict")),
             "Confidence": "%d%%" % c["ai_summary"]["confidence"], "Submitted (UTC)": c["created_ts"]} for i, c in enumerate(queue)]
    wide(st.dataframe, pd.DataFrame(rows), hide_index=True)
    labels = {"%s - %s [%s] (%s)" % (c["case_id"], c["patient_name"], c["ai_summary"]["urgency"], c["status"]): c["case_id"] for c in queue}
    choice = st.selectbox("Open a case:", list(labels))
    c = next(x for x in queue if x["case_id"] == labels[choice])
    log_view(c["case_id"])
    if c["status"] == "Submitted":
        st.info("This case has not been verified by a nurse yet.")
    pi = c["patient_input"]
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("##### 📥 Patient intake & nurse notes")
            ptext("Patient:", "%s %s · %s %s · %s · prefers %s" % (pi.get("title", ""), pi.get("name", ""), pi.get("age"), pi.get("age_unit"), pi.get("sex"), pi.get("lang")))
            ptext("History:", pi.get("history") or "None stated")
            ptext("Allergies / medicines:", "%s / %s" % (pi.get("allergies") or "not provided", pi.get("medications") or "not provided"))
            no = c["nurse_output"]
            if no:
                ptext("Nurse verification:", no.get("triage_notes", ""))
                ptext("Nurse re-checked vitals:", eng.vitals_text(no.get("vitals_recheck", {})) or "none entered")
            else:
                st.caption("No nurse verification yet.")
            render_sources(c, "d_" + c["case_id"])
    with right:
        with st.container(border=True):
            st.markdown("##### 🧠 AI extraction, flags and evidence")
            render_ai_panel(c)
    if c["status"] == "Reviewed":
        do = c["doctor_output"]
        with st.container(border=True):
            st.markdown("##### ✅ Reviewed by %s" % esc(do.get("reviewer", "")), unsafe_allow_html=True)
            ptext("Final triage level:", "%s (AI: %s)%s" % (do.get("final_urgency"), do.get("ai_urgency"), (" - override: " + do["override_reason"]) if do.get("override_reason") else ""))
            ptext("Clinical note:", do.get("referral_note", ""))
            ptext("Patient message:", do.get("patient_message", ""))
    else:
        doctor_actions(c)
    with st.expander("🧾 Audit trail for this case"):
        wide(st.dataframe, pd.DataFrame(db.get_audit(c["case_id"])), hide_index=True)


def dietician_view():
    """Role-based access: dieticians only see reviewed cases, minimal fields, and can edit only the diet chart."""
    st.markdown("### 🍱 Dietician workspace")
    cases = db.list_cases(statuses=["Reviewed"])
    if not cases:
        st.info("No reviewed cases yet. A physician must approve a case before a diet chart can be added.")
        return
    labels = {"%s - %s" % (c["case_id"], c["patient_name"]): c["case_id"] for c in cases}
    c = next(x for x in cases if x["case_id"] == labels[st.selectbox("Open a case:", list(labels))])
    log_view(c["case_id"])
    pi, do = c["patient_input"], c["doctor_output"]
    with st.container(border=True):
        ptext("Patient:", "%s · %s %s · %s" % (pi.get("name"), pi.get("age"), pi.get("age_unit"), pi.get("sex")))
        ptext("History:", pi.get("history") or "None stated")
        st.caption("Limited view: symptoms, vitals and physician notes are hidden from this role.")
    cid = c["case_id"]
    for k, name in zip("maev", ("Morning", "Afternoon", "Evening", "AVOID")):
        ss.setdefault("diet_%s_%s" % (k, cid), (do.get("diet_chart") or {}).get(name, ""))
    with st.container(border=True):
        diet = _diet_inputs(cid)
        if wide(st.button, "💾 Save diet chart", key="sd_%s" % cid, type="primary"):
            data = data_of(c)
            data["doctor_output"] = {**do, "diet_chart": diet}
            try:
                db.update_case(cid, data, c["status"], c["urgency"], c["news2"], expected_version=c["version"])
            except db.ConflictError as e:
                st.error(str(e))
                return
            db.audit(ss.username, ss.role, "diet_chart_updated", cid)
            flash("Diet chart saved.")
            st.rerun()


# =====================================================================
# ROUTER + FOOTER
# =====================================================================
if ss.role is None:
    login_screen()
elif ss.role == "Patient View":
    patient_view()
elif ss.role == "Emergency Nurse Node":
    nurse_view()
elif ss.role == "Attending Doctor (MD) View":
    doctor_view()

st.markdown("<br><hr>", unsafe_allow_html=True)
f1, f2 = st.columns(2)
with f1:
    st.markdown("<p style='color:#e53e3e;font-size:12px;font-weight:bold;line-height:1.4;margin:0;'>⚠️ This system does NOT provide diagnosis. It supports triage only. The final decision is made by a qualified medical professional.</p>", unsafe_allow_html=True)
with f2:
    st.markdown("<p style='text-align:right;font-size:11px;color:#718096;margin:0;'>🔒 Identifiers auto-redacted · uploaded files &amp; audio never stored · role-based access · every AI and human action is logged (see 'Audit trail' per case)</p>", unsafe_allow_html=True)
