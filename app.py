"""Giao diện Streamlit. Chạy: python -m streamlit run app.py"""
import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predictor import MODEL_INFO, Predictor  # noqa: E402

METRICS_FILE = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --ink:#eaf0ff; --muted:#9aa8c5; --line:rgba(148,163,184,.18); }
.stApp { background: radial-gradient(circle at 15% 0%, #18264a 0, #0a1020 32%, #070b14 72%); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: rgba(8,13,26,.92); border-right:1px solid var(--line); }
.hero { padding: 1.7rem 1.8rem; border:1px solid var(--line); border-radius:24px;
        background:linear-gradient(135deg,rgba(59,130,246,.16),rgba(139,92,246,.08));
        box-shadow:0 24px 70px rgba(0,0,0,.24); margin-bottom:1.2rem; }
.eyebrow { color:#7dd3fc; font-size:.78rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { color:#f8fbff; font-size:2.25rem; margin:.32rem 0 .25rem; letter-spacing:-.04em; }
.hero p { color:#aebbd3; max-width:720px; margin:0; }
.glass { border:1px solid var(--line); border-radius:20px; padding:1.15rem;
         background:rgba(14,22,40,.72); box-shadow:0 18px 50px rgba(0,0,0,.18); }
.verdict { border-radius:18px; padding:1.2rem 1.25rem; min-height:142px; border:1px solid var(--line); }
.verdict.real { background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(6,78,59,.18)); }
.verdict.fake { background:linear-gradient(135deg,rgba(244,63,94,.18),rgba(127,29,29,.18)); }
.verdict .label { font-size:.78rem; color:#b8c3d9; font-weight:700; letter-spacing:.12em; }
.verdict .value { font-size:2rem; color:#fff; font-weight:850; margin:.15rem 0; }
.verdict .sub { color:#b4c0d6; font-size:.9rem; }
.ring { width:142px; height:142px; border-radius:50%; display:grid; place-items:center; margin:auto;
        background:conic-gradient(var(--ring) calc(var(--score)*1%), rgba(148,163,184,.16) 0); position:relative; }
.ring:before { content:""; position:absolute; inset:12px; border-radius:50%; background:#0c1425; }
.ring-inner { position:relative; text-align:center; color:white; line-height:1; }
.ring-score { font-size:1.65rem; font-weight:850; }
.ring-caption { font-size:.72rem; color:#9caccc; margin-top:.38rem; }
.model-card { border:1px solid var(--line); border-radius:16px; padding:1rem; background:rgba(15,23,42,.72); }
.model-name { color:#f5f8ff; font-weight:750; }
.model-meta { color:#94a3bd; font-size:.82rem; margin-top:.18rem; }
.pill { display:inline-block; padding:.22rem .58rem; border-radius:999px; font-size:.72rem; font-weight:800; }
.pill-real { color:#6ee7b7; background:rgba(16,185,129,.14); }
.pill-fake { color:#fda4af; background:rgba(244,63,94,.14); }
.best { color:#fde68a; background:rgba(245,158,11,.13); border:1px solid rgba(245,158,11,.25); }
[data-testid="stMetric"] { padding:.8rem 1rem; border:1px solid var(--line); border-radius:15px; background:rgba(15,23,42,.58); }
.stButton > button { border-radius:12px; min-height:46px; font-weight:750; }
.stTextInput input, .stTextArea textarea { border-radius:13px !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.small-note { color:#8492ad; font-size:.8rem; text-align:center; margin-top:1rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def test_f1(key: str, metrics: dict) -> float | None:
    value = metrics.get(key, {}).get("test", {}).get("f1")
    return float(value) if isinstance(value, (int, float)) else None


def best_available_model(available: list[str], metrics: dict) -> str | None:
    ranked = [(test_f1(key, metrics), key) for key in available]
    ranked = [(score, key) for score, key in ranked if score is not None]
    return max(ranked)[1] if ranked else (available[0] if available else None)


def ring(score: float, label: str) -> str:
    color = "#34d399" if label == "REAL" else "#fb7185"
    return f"""
    <div class="ring" style="--score:{score * 100:.1f};--ring:{color}">
      <div class="ring-inner"><div class="ring-score">{score:.0%}</div>
      <div class="ring-caption">ĐỘ TIN CẬY</div></div>
    </div>"""


def metrics_table(metrics: dict, available: list[str]) -> pd.DataFrame:
    rows = []
    for key in available:
        res = metrics.get(key, {})
        test = res.get("test", {})
        rows.append({
            "Mô hình": MODEL_INFO.get(key, key),
            "Accuracy": test.get("accuracy"),
            "Precision": test.get("precision"),
            "Recall": test.get("recall"),
            "F1": test.get("f1"),
            "Train (giây)": res.get("train_seconds"),
        })
    return pd.DataFrame(rows)


predictor = get_predictor()
available = predictor.available()
metrics = load_metrics()
best_key = best_available_model(available, metrics)

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Machine Learning • News Intelligence</div>
  <h1>TruthLens AI</h1>
  <p>Phân tích tín hiệu ngôn ngữ trong bài báo tiếng Anh và đối chiếu kết quả giữa các mô hình học máy.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Trung tâm phân tích")
    if available:
        default_index = available.index(best_key) if best_key in available else 0
        chosen = st.selectbox(
            "Mô hình chính",
            available,
            index=default_index,
            format_func=lambda key: MODEL_INFO[key],
        )
        chosen_f1 = test_f1(chosen, metrics)
        if chosen == best_key and chosen_f1 is not None:
            st.markdown(
                f'<span class="pill best">★ F1 test cao nhất: {chosen_f1:.1%}</span>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption(f"{len(available)} mô hình sẵn sàng")
        compare_all = st.toggle("So sánh tất cả mô hình", value=True)
    else:
        chosen = None
        compare_all = False
        st.error("Chưa tìm thấy model đã train.")
        st.code("python src/train_baseline.py", language="powershell")

    st.divider()
    st.markdown("**Quy ước**")
    st.markdown("🟢 REAL  ·  có xu hướng là tin thật")
    st.markdown("🔴 FAKE  ·  có xu hướng là tin giả")

predict_tab, benchmark_tab = st.tabs(["✦ Phân tích", "◫ Hiệu năng mô hình"])

with predict_tab:
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown("#### Nội dung cần kiểm tra")
        title = st.text_input("Tiêu đề", placeholder="Nhập tiêu đề bài báo tiếng Anh...")
        body = st.text_area("Nội dung", height=310, placeholder="Dán nội dung bài báo tại đây...")
        analyze = st.button("Phân tích ngay  →", type="primary", use_container_width=True, disabled=chosen is None)

    with right:
        st.markdown("#### Tổng quan")
        if not analyze:
            st.markdown(
                """
                <div class="glass" style="min-height:355px;display:grid;place-items:center;text-align:center">
                  <div><div style="font-size:3rem;color:#60a5fa">◉</div>
                  <div style="color:#f3f7ff;font-size:1.15rem;font-weight:750;margin-top:.5rem">Sẵn sàng phân tích</div>
                  <div style="color:#8fa0bd;margin-top:.35rem">Kết quả, độ tin cậy và so sánh model<br>sẽ xuất hiện tại đây.</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if analyze:
        if not title.strip() and not body.strip():
            st.warning("Bạn chưa nhập nội dung cần phân tích.")
        else:
            keys = available if compare_all else [chosen]
            results, errors = {}, {}
            with st.spinner("Đang phân tích tín hiệu ngôn ngữ..."):
                for key in keys:
                    try:
                        results[key] = predictor.predict(key, title, body)
                    except Exception as exc:
                        errors[key] = str(exc)
                if chosen not in results:
                    try:
                        results[chosen] = predictor.predict(chosen, title, body)
                    except Exception as exc:
                        errors[chosen] = str(exc)

            if chosen in results:
                result = results[chosen]
                with right:
                    verdict_class = "real" if result["label"] == "REAL" else "fake"
                    safe_name = html.escape(result["name"])
                    st.markdown(
                        f"""
                        <div class="verdict {verdict_class}">
                          <div class="label">KẾT LUẬN TỪ MÔ HÌNH CHÍNH</div>
                          <div class="value">{result['label']}</div>
                          <div class="sub">{safe_name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    r1, r2 = st.columns([.9, 1.1])
                    with r1:
                        st.markdown(ring(result["confidence"], result["label"]), unsafe_allow_html=True)
                    with r2:
                        st.metric("Khả năng FAKE", f"{result['fake_prob']:.1%}")
                        st.metric("Khả năng REAL", f"{result['real_prob']:.1%}")
                    st.caption(f"Phản hồi trong {result['ms']:.1f} ms · {result['words']:,} từ sau làm sạch")
            elif errors:
                with right:
                    st.error("Không thể phân tích bằng mô hình đã chọn: " + errors.get(chosen, "Lỗi không xác định"))

            if len(results) > 1:
                st.divider()
                st.markdown("### So sánh mô hình")
                labels = [r["label"] for r in results.values()]
                agreement = max(labels.count("REAL"), labels.count("FAKE")) / len(labels)
                c1, c2, c3 = st.columns(3)
                c1.metric("Mức đồng thuận", f"{agreement:.0%}")
                c2.metric("Dự đoán REAL", labels.count("REAL"))
                c3.metric("Dự đoán FAKE", labels.count("FAKE"))

                cols = st.columns(min(len(results), 4))
                for index, (key, item) in enumerate(results.items()):
                    col = cols[index % len(cols)]
                    score = test_f1(key, metrics)
                    badge = '<span class="pill best">★ Test tốt nhất</span>' if key == best_key and score is not None else ""
                    pill_class = "pill-real" if item["label"] == "REAL" else "pill-fake"
                    f1_text = f"F1 test {score:.1%}" if score is not None else "Chưa có metric test"
                    with col:
                        st.markdown(
                            f"""
                            <div class="model-card">
                              <div class="model-name">{html.escape(item['name'])}</div>
                              <div style="margin:.6rem 0"><span class="pill {pill_class}">{item['label']}</span> {badge}</div>
                              <div style="color:#e7edfb;font-size:1.35rem;font-weight:800">{item['confidence']:.0%}</div>
                              <div class="model-meta">Độ tin cậy dự đoán · {f1_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                table_rows = []
                for key, item in results.items():
                    table_rows.append({
                        "Mô hình": item["name"],
                        "Kết luận": item["label"],
                        "P(FAKE)": item["fake_prob"],
                        "Độ tin cậy": item["confidence"],
                        "F1 test": test_f1(key, metrics),
                        "Thời gian (ms)": item["ms"],
                    })
                compare_df = pd.DataFrame(table_rows)
                st.dataframe(
                    compare_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "P(FAKE)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                        "Độ tin cậy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                        "F1 test": st.column_config.NumberColumn(format="%.3f"),
                        "Thời gian (ms)": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
                if agreement < 1:
                    st.info("Các mô hình chưa đồng thuận. Ưu tiên mô hình có F1 test cao hơn và kiểm tra thêm nguồn tin.")

            for key, message in errors.items():
                st.caption(f"{MODEL_INFO.get(key, key)} không phản hồi: {message}")

with benchmark_tab:
    st.markdown("### Bảng xếp hạng trên tập test")
    table = metrics_table(metrics, available)
    if table.empty or table["F1"].isna().all():
        st.info("Số liệu sẽ xuất hiện sau khi các script train hoàn tất.")
    else:
        table = table.sort_values("F1", ascending=False, na_position="last").reset_index(drop=True)
        table.insert(0, "Hạng", range(1, len(table) + 1))
        top = table.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Mô hình dẫn đầu", top["Mô hình"])
        b.metric("F1 test", f"{top['F1']:.1%}")
        c.metric("Accuracy", f"{top['Accuracy']:.1%}")
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                "Train (giây)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.markdown(
            '<div class="small-note">“Mô hình đáng tin nhất” được xếp theo F1 trên tập test, không chỉ theo confidence của một dự đoán.</div>',
            unsafe_allow_html=True,
        )
