# -*- coding: utf-8 -*-
"""
env_mode_dashboard.py
Streamlit app for environmental data analysis with 3 main modes:
1) Temperature
2) Humidity
3) Light  (Intensity / Photoperiod / Daily Cumulative (DLI-like))

Assumptions
- Data contain a time-like column (auto-detected) and numeric sensor columns.
- Light "on" is detected by a threshold (default > 0). You can change it.
- Sampling interval is inferred from median delta between timestamps (works with 10-min data).

Author: generated collaboratively
"""

import io
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="환경 데이터 3모드 분석", layout="wide")
st.title("🌿 환경 데이터 3모드 분석 (업로드 기반)")

# ------------------------------
# Helpers
# ------------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_"," ").replace("-"," ")

def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    cand = [c for c in df.columns if any(k in _norm(c) for k in ["time","date","timestamp","datetime","일시","시각","측정"])]
    scores = []
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            scores.append((c, parsed.notna().mean()))
        except Exception:
            scores.append((c, 0.0))
    def key(x): return (x[0] in cand, x[1])
    scores.sort(key=key, reverse=True)
    return scores[0][0] if scores and scores[0][1] > 0.5 else None

@st.cache_data(show_spinner=False)
def read_any(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    ext = str(file_name).lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        # Excel: merge all sheets
        xlf = pd.ExcelFile(io.BytesIO(file_bytes))
        frames = []
        for s in xlf.sheet_names:
            try:
                df = xlf.parse(s)
                df["__Sheet__"] = s
                frames.append(df)
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def coerce_time(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    out = df.copy()
    out["Timestamp"] = pd.to_datetime(out[tcol], errors="coerce", infer_datetime_format=True)
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return out

def infer_interval_seconds(ts: pd.Series) -> Optional[float]:
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return None
    return float(diffs.dt.total_seconds().median())

def choose_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def bar_with_error(mean_val: float, std_val: float, title: str, y_label: str = ""):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[title],
        y=[mean_val],
        error_y=dict(type='data', array=[std_val]),
        name="Mean ± SD"
    ))
    fig.update_layout(yaxis_title=y_label, showlegend=False)
    return fig

# Photoperiod detection: first-on & last-off per day
def photoperiod_table(df: pd.DataFrame, light_col: str, on_thr: float, interval_sec: Optional[float]) -> pd.DataFrame:
    tmp = df[["Timestamp", light_col]].dropna().copy()
    tmp["date"] = tmp["Timestamp"].dt.date
    tmp["is_on"] = tmp[light_col] > on_thr
    day_list = []
    step_sec = interval_sec if interval_sec is not None else 600.0  # default 10min
    for d, sub in tmp.groupby("date"):
        if not sub["is_on"].any():
            day_list.append({"Date": d, "On Time": None, "Off Time": None, "Photoperiod (h)": 0.0, "Segments": 0})
            continue
        # segments detection via run-length on is_on
        segs = []
        cur_on = None
        for t, flag in zip(sub["Timestamp"].tolist(), sub["is_on"].tolist()):
            if flag and cur_on is None:
                cur_on = t
            elif not flag and cur_on is not None:
                segs.append((cur_on, t))
                cur_on = None
        if cur_on is not None:
            # closing open segment; approximate off at last time + one step
            last_t = sub["Timestamp"].iloc[-1]
            off_t = last_t + pd.to_timedelta(step_sec, unit="s")
            segs.append((cur_on, off_t))
        first_on = segs[0][0] if segs else None
        last_off = segs[-1][1] if segs else None
        total_h = sum([(b - a).total_seconds() for a, b in segs]) / 3600.0 if segs else 0.0
        day_list.append({
            "Date": d,
            "On Time": first_on,
            "Off Time": last_off,
            "Photoperiod (h)": round(total_h, 3),
            "Segments": len(segs)
        })
    return pd.DataFrame(day_list).sort_values("Date")

def daily_cumulative(df: pd.DataFrame, light_col: str, interval_sec: Optional[float], only_when_on: bool, on_thr: float) -> pd.DataFrame:
    tmp = df[["Timestamp", light_col]].dropna().copy()
    if only_when_on:
        tmp = tmp[tmp[light_col] > on_thr]
    tmp["date"] = tmp["Timestamp"].dt.date
    step = interval_sec if interval_sec is not None else 600.0  # seconds
    # integral = sum(light * dt)
    daily = tmp.groupby("date")[light_col].sum() * step
    daily = daily.rename("DailyIntegral")
    out = daily.reset_index().rename(columns={"date":"Date"})
    return out

# ------------------------------
# Sidebar: Upload & mapping
# ------------------------------
with st.sidebar:
    st.header("📁 파일 업로드")
    up = st.file_uploader("환경 데이터 업로드 (xlsx/xls/csv)", type=["xlsx","xls","csv"])

if up is None:
    st.info("왼쪽에서 파일을 업로드하면 분석을 시작합니다.")
    st.stop()

df_raw = read_any(up.getvalue(), up.name)
if df_raw.empty:
    st.warning("업로드된 파일에서 데이터를 읽을 수 없습니다. 파일 내용을 확인하세요.")
    st.stop()

# Time column detection & coercion
tcol_auto = detect_time_column(df_raw)
time_col = st.sidebar.selectbox("시간 컬럼 선택", options=["<자동 탐지>"] + df_raw.columns.tolist(), index=0)
if time_col == "<자동 탐지>":
    time_col = tcol_auto
if time_col is None:
    st.warning("시간 컬럼을 자동으로 찾지 못했습니다. 사이드바에서 직접 선택해 주세요.")
    st.stop()

df = coerce_time(df_raw, time_col)

# Numeric columns & mapping
exclude_cols = ["Timestamp","__Sheet__"]
num_cols = choose_numeric_columns(df, exclude=exclude_cols)
st.sidebar.markdown("---")
st.sidebar.header("🧭 변수 매핑")
temp_col = st.sidebar.selectbox("온도 컬럼", options=["<없음>"] + num_cols, index=0)
rh_col   = st.sidebar.selectbox("습도 컬럼", options=["<없음>"] + num_cols, index=0)
light_col= st.sidebar.selectbox("광 컬럼", options=["<없음>"] + num_cols, index=0)

# Date range
st.sidebar.markdown("---")
st.sidebar.header("⏱️ 기간 필터")
min_t, max_t = df["Timestamp"].min(), df["Timestamp"].max()
sel_range = st.sidebar.slider("기간 선택", value=(min_t, max_t), min_value=min_t, max_value=max_t, format="YYYY-MM-DD HH:mm")

df = df[(df["Timestamp"] >= sel_range[0]) & (df["Timestamp"] <= sel_range[1])].copy()
if df.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

# Sampling interval
interval_sec = infer_interval_seconds(df["Timestamp"])

# ------------------------------
# Main Mode
# ------------------------------
mode = st.selectbox("분석 모드 선택", ["온도", "습도", "광"], index=0)

# ---------- 온도 ----------
if mode == "온도":
    if temp_col == "<없음>":
        st.info("사이드바에서 온도 컬럼을 지정하세요.")
        st.stop()
    series = df[["Timestamp", temp_col]].dropna()
    mean_val = float(series[temp_col].mean())
    std_val  = float(series[temp_col].std())
    c1, c2 = st.columns([1,2])
    with c1:
        st.subheader("📊 평균±표준편차 (막대)")
        st.plotly_chart(bar_with_error(mean_val, std_val, "Temperature", y_label=temp_col), use_container_width=True)
        st.caption(f"기간 평균: **{mean_val:.3f}**, 표준편차: **{std_val:.3f}**")
    with c2:
        st.subheader("📈 시계열 그래프")
        fig = px.line(series, x="Timestamp", y=temp_col, title=f"{temp_col} 시간 변화")
        st.plotly_chart(fig, use_container_width=True)

# ---------- 습도 ----------
elif mode == "습도":
    if rh_col == "<없음>":
        st.info("사이드바에서 습도 컬럼을 지정하세요.")
        st.stop()
    series = df[["Timestamp", rh_col]].dropna()
    mean_val = float(series[rh_col].mean())
    std_val  = float(series[rh_col].std())
    c1, c2 = st.columns([1,2])
    with c1:
        st.subheader("📊 평균±표준편차 (막대)")
        st.plotly_chart(bar_with_error(mean_val, std_val, "Humidity", y_label=rh_col), use_container_width=True)
        st.caption(f"기간 평균: **{mean_val:.3f}**, 표준편차: **{std_val:.3f}**")
    with c2:
        st.subheader("📈 시계열 그래프")
        fig = px.line(series, x="Timestamp", y=rh_col, title=f"{rh_col} 시간 변화")
        st.plotly_chart(fig, use_container_width=True)

# ---------- 광 ----------
else:
    if light_col == "<없음>":
        st.info("사이드바에서 광 컬럼을 지정하세요.")
        st.stop()

    st.markdown("### 🔦 광 분석 서브모드")
    submode = st.radio("서브모드 선택", ["1) 광도", "2) 광주기", "3) 적산광도"], horizontal=True)

    st.sidebar.markdown("---")
    st.sidebar.header("💡 광 파라미터")
    on_thr = st.sidebar.number_input("Light ON 임계값(초과)", value=0.0, step=0.1, help="이 값 초과를 켜짐으로 간주")
    only_on_for_intensity = True  # for mode 1
    convert_to_dli = st.sidebar.checkbox("PPFD(μmol m⁻² s⁻¹) → DLI(몰 m⁻² 일⁻¹) 변환", value=False, help="적산 시 1e6으로 나눔")

    if submode.startswith("1"):
        # Intensity (lights on only)
        sub = df[["Timestamp", light_col]].dropna()
        sub = sub[sub[light_col] > on_thr]
        if sub.empty:
            st.info("임계값 초과 구간(켜짐)이 없습니다. 임계값을 조정해 보세요.")
        else:
            mean_val = float(sub[light_col].mean())
            std_val  = float(sub[light_col].std())
            c1, c2 = st.columns([1,2])
            with c1:
                st.subheader("📊 켜짐 구간 광도 평균±표준편차")
                st.plotly_chart(bar_with_error(mean_val, std_val, "Light Intensity", y_label=light_col), use_container_width=True)
                st.caption(f"켜짐 구간 평균: **{mean_val:.3f}**, 표준편차: **{std_val:.3f}** (임계값 {on_thr} 초과 데이터만 사용)")
            with c2:
                st.subheader("📈 시계열(켜짐 구간만)")
                fig = px.line(sub, x="Timestamp", y=light_col, title=f"{light_col} 시간 변화 (ON만)")
                st.plotly_chart(fig, use_container_width=True)

    elif submode.startswith("2"):
        # Photoperiod table
        st.subheader("🕑 광주기(Photoperiod) 일자별 표")
        table = photoperiod_table(df, light_col, on_thr=on_thr, interval_sec=interval_sec)
        st.dataframe(table, use_container_width=True)
        # Download
        csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button("광주기 표 CSV 다운로드", data=csv_bytes, file_name="photoperiod_table.csv", mime="text/csv")
        if interval_sec is not None:
            st.caption(f"추정 샘플링 간격: **{interval_sec:.0f}초**")

    else:
        # Daily cumulative (DLI-like)
        st.subheader("☀️ 적산광도")
        day_int = daily_cumulative(df, light_col, interval_sec=interval_sec, only_when_on=False, on_thr=on_thr)
        if day_int.empty:
            st.info("적산 결과가 없습니다.")
        else:
            # Optional conversion to DLI (if PPFD provided)
            if convert_to_dli:
                day_int["DailyIntegral"] = day_int["DailyIntegral"] / 1_000_000.0
                unit_label = "DLI (mol m⁻² d⁻¹)"
            else:
                unit_label = f"{light_col}×s (적분값)"
            day_int = day_int.rename(columns={"DailyIntegral": unit_label})

            # Per-day plot
            fig = px.bar(day_int, x="Date", y=unit_label, title="일별 적산광도", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

            # Mean ± SD across days
            mean_val = float(day_int[unit_label].mean())
            std_val  = float(day_int[unit_label].std())
            st.subheader("📊 전체 기간의 평균 일일 적산광도 (Mean ± SD)")
            st.plotly_chart(bar_with_error(mean_val, std_val, "Daily Cumulative", y_label=unit_label), use_container_width=True)
            st.caption(f"평균: **{mean_val:.3f}**, 표준편차: **{std_val:.3f}**")

            # Download
            csv_bytes = day_int.to_csv(index=False).encode("utf-8-sig")
            st.download_button("일별 적산광도 CSV 다운로드", data=csv_bytes, file_name="daily_cumulative_light.csv", mime="text/csv")
            if interval_sec is not None:
                st.caption(f"추정 샘플링 간격: **{interval_sec:.0f}초**")
