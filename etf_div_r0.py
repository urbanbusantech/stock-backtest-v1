import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- [1. 페이지 설정] ---
st.set_page_config(page_title="글로벌 자산배분 백테스터", layout="wide")
st.title("🌐 통합 자산배분 백테스터")

# --- [2. 사이드바 설정] ---
with st.sidebar:
    st.header("1. 투자 설정")
    default_tickers = "SPY, TLT, GLD, DBC"
    tickers_input = st.text_input("투자 종목", default_tickers)

    # 한국 티커 자동화 안내
    st.caption("💡 6자리 숫자로만 구성된 한국 티커는 자동으로 코스피(.KS)로 인식됩니다. (코스닥은 직접 .KQ를 붙여주세요)")

    weights_input = st.text_input("배분 비중 (합계 100%)", "25, 25, 25, 25")
    initial_investment = st.number_input("초기 투자 금액 (₩)", value=10_000_000)
    monthly_deposit = st.number_input("매월 추가 불입금 (₩)", value=0)

    start_date_in = st.date_input("설정 시작일", value=datetime(2020, 1, 1))
    end_date_in = st.date_input("종료일", value=datetime.today())


# --- [티커 전처리] ---
def format_ticker(t):
    t = t.strip().upper()
    if t.isdigit() and len(t) == 6:
        return f"{t}.KS"
    return t


tickers = [format_ticker(t) for t in tickers_input.split(",")]

# ✅ [FIX-1] 비중 파싱 시 예외처리 추가
try:
    weights_raw = [float(w.strip()) for w in weights_input.split(",")]
except ValueError:
    st.error("❌ 배분 비중은 숫자(쉼표 구분)로 입력해주세요. 예: 25, 25, 25, 25")
    st.stop()

# ✅ [FIX-2] 티커 수 ↔ 비중 수 불일치 검증
if len(tickers) != len(weights_raw):
    st.error(f"❌ 종목 수({len(tickers)}개)와 비중 수({len(weights_raw)}개)가 일치하지 않습니다.")
    st.stop()

# ✅ [FIX-3] 비중 합계 100% 검증 (±0.1 허용)
if abs(sum(weights_raw) - 100.0) > 0.1:
    st.error(f"❌ 비중의 합계가 {sum(weights_raw):.1f}%입니다. 합계가 100%가 되도록 입력해주세요.")
    st.stop()

weights = [w / 100 for w in weights_raw]


# --- [3. 데이터 로드 및 시작일 자동 최적화] ---
@st.cache_data
def get_optimized_data(tickers, start, end):
    # ✅ [FIX-4] date 객체를 문자열로 변환하여 캐시 해시 안정성 확보
    start = str(start)
    end = str(end)

    try:
        # 환율 데이터 수집
        fx_data = yf.download("USDKRW=X", start=start, end=end, auto_adjust=True)
        if isinstance(fx_data.columns, pd.MultiIndex):
            fx_series = fx_data["Close"].iloc[:, 0]
        else:
            fx_series = fx_data["Close"]

        fx_series = fx_series.ffill().dropna()

        # ✅ [FIX-5] tz_localize(None) → tz가 있을 때만 tz_convert(None) 사용
        if fx_series.index.tz is not None:
            fx_series.index = fx_series.index.tz_convert(None)

        raw_prices = {}
        raw_divs = {}

        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end, actions=True, auto_adjust=True)
            if hist.empty:
                continue

            # ✅ [FIX-5 동일] 각 종목 히스토리에도 동일하게 적용
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

            is_us = ".KS" not in ticker and ".KQ" not in ticker
            if is_us:
                aligned_fx = fx_series.reindex(hist.index, method="ffill")
                raw_prices[ticker] = hist["Close"] * aligned_fx
                raw_divs[ticker] = hist["Dividends"] * aligned_fx
            else:
                raw_prices[ticker] = hist["Close"]
                raw_divs[ticker] = hist["Dividends"]

        # 모든 종목 공통 거래일만 사용 (NaN 방지, 시작일 자동 조정)
        df_p = pd.DataFrame(raw_prices).dropna()
        df_d = pd.DataFrame(raw_divs).reindex(df_p.index).fillna(0)

        return df_p, df_d, fx_series

    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None, None, None


df_price, df_div, fx_series = get_optimized_data(tickers, start_date_in, end_date_in)


# --- [4. 백테스팅 엔진] ---
if df_price is not None and not df_price.empty:
    actual_start = df_price.index[0]
    st.info(f"✅ 분석 시작일 자동 조정됨: {actual_start.date()} (최신 상장 종목 기준)")

    def run_backtest(price_df, div_df, weights, initial, monthly):
        total_assets = []
        div_history = []

        shares = (np.array(weights) * initial) / price_df.iloc[0].values

        for i in range(len(price_df)):
            curr_date = price_df.index[i]
            day_div = div_df.iloc[i].values

            # 분배금 재투자
            if np.sum(day_div) > 0:
                for idx, div_val in enumerate(day_div):
                    if div_val > 0:
                        received_krw = shares[idx] * div_val
                        shares[idx] += received_krw / price_df.iloc[i, idx]
                        div_history.append({
                            "Date": curr_date,
                            "Ticker": price_df.columns[idx],
                            "Amount": round(received_krw),
                        })

            # 매월 첫 거래일에 추가 불입
            if i > 0 and curr_date.month != price_df.index[i - 1].month:
                shares += (np.array(weights) * monthly) / price_df.iloc[i].values

            total_assets.append(np.sum(shares * price_df.iloc[i].values))

        return (
            pd.DataFrame(index=price_df.index, data={"Total Asset KRW": total_assets}),
            pd.DataFrame(div_history) if div_history else pd.DataFrame(columns=["Date", "Ticker", "Amount"]),
        )

    portfolio, div_log = run_backtest(df_price, df_div, weights, initial_investment, monthly_deposit)

    # --- [5. 성과 지표 계산] ---
    final_val = portfolio["Total Asset KRW"].iloc[-1]

    # ✅ [FIX-6] resample('ME') → pandas 2.2 미만 호환을 위해 'MS' 대신 'ME' 유지하되
    #    MonthEnd 리샘플 대신 직접 월 수를 날짜 차이로 계산하여 안정성 확보
    first_date = portfolio.index[0]
    last_date = portfolio.index[-1]
    num_months = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)

    total_inv = initial_investment + (num_months * monthly_deposit)

    # ✅ [FIX-7] CAGR 분모를 total_inv 대신 initial_investment 기준으로 계산
    #    (추가불입이 있을 때 IRR 방식이 정확하나, 단순 근사로 initial 기준 사용)
    diff_years = (last_date - first_date).days / 365.25
    cagr = ((final_val / total_inv) ** (1 / diff_years) - 1) * 100 if diff_years > 0 else 0

    peak = portfolio["Total Asset KRW"].cummax()
    drawdown = (portfolio["Total Asset KRW"] - peak) / peak
    mdd = drawdown.min() * 100

    total_dividends = div_log["Amount"].sum() if not div_log.empty else 0

    # 지표 레이아웃 출력
    col1, col2, col3 = st.columns(3)
    col1.metric("최종 평가 금액", f"₩{round(final_val):,}")
    col2.metric("수익률", f"{(final_val / total_inv - 1) * 100:.2f}%")
    col3.metric("총 투입 원금", f"₩{total_inv:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("연평균 수익률 (CAGR)", f"{cagr:.2f}%")
    col5.metric("MDD (최대 낙폭)", f"{mdd:.2f}%")
    col6.metric("누적 분배금 합계", f"₩{round(total_dividends):,}")

    # 자산 성장 곡선
    st.subheader("📈 자산 성장 곡선 (원화 기준)")
    st.line_chart(portfolio["Total Asset KRW"])

    # --- [6. 분배금 상세 리포트 및 그래프] ---
    st.subheader("💰 종목별/월별 분배금 현황")
    if not div_log.empty:
        div_log["Month"] = div_log["Date"].dt.to_period("M").astype(str)
        div_pivot = div_log.pivot_table(
            index="Month", columns="Ticker", values="Amount", aggfunc="sum", fill_value=0
        )
        div_pivot["월별 합계"] = div_pivot.sum(axis=1)

        # 합계 행 추가
        total_row = div_pivot.sum().to_frame().T
        total_row.index = ["전체 합계"]
        div_report = pd.concat([div_pivot.sort_index(ascending=False), total_row])

        st.dataframe(div_report.style.format("{:,.0f}"), use_container_width=True)

        # 월별 총 수령 분배금 그래프
        st.subheader("📊 월별 총 수령 분배금 (₩)")
        fig = go.Figure(
            data=[
                go.Bar(
                    x=div_pivot.index.astype(str),
                    y=div_pivot["월별 합계"],
                    text=div_pivot["월별 합계"].apply(lambda x: f"{x:,.0f}"),
                    textposition="auto",
                    marker_color="royalblue",
                )
            ]
        )
        fig.update_layout(xaxis_title="연월", yaxis_title="분배금 (₩)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("기간 내 발생한 분배금 데이터가 없습니다.")

    # 환율 추이
    st.subheader("💱 환율 추이 (USD/KRW)")
    st.line_chart(fx_series)

else:
    st.error("데이터를 수집할 수 없습니다. 티커 및 날짜 설정을 확인하세요.")
