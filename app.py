
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.naive_bayes import GaussianNB
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

import matplotlib
import platform


# --- 한글 폰트 설정 ---
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rc('axes', unicode_minus=False)

st.set_page_config(layout="wide")
st.title("스마트팜 수확량 + 생육 예측 XAI 통합 대시보드")

# --- 파일 업로드 ---
sensor_file = st.file_uploader("환경센서 데이터 업로드 (CSV)", type=["csv"])
yield_file = st.file_uploader("수확/생육 데이터 업로드 (CSV)", type=["csv"])

if sensor_file and yield_file:
    sensor_df = pd.read_csv(sensor_file)
    yield_df = pd.read_csv(yield_file)

    st.subheader("환경센서 데이터")
    st.dataframe(sensor_df.head())
    st.subheader("수확/생육 데이터")
    st.dataframe(yield_df.head())

    # --- 컬럼 선택 (환경센서: 가로 배치) ---
    st.subheader("컬럼 선택")

    # ✅ 환경센서 컬럼 선택 - 한 줄 가로 배치
    st.markdown("**환경 센서 데이터 컬럼 선택**")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        date_col_sensor = st.selectbox("날짜시간", sensor_df.columns)
    with col2:
        temp_col = st.selectbox("온도", sensor_df.columns)
    with col3:
        hum_col = st.selectbox("습도", sensor_df.columns)
    with col4:
        co2_col = st.selectbox("CO₂", sensor_df.columns)
    with col5:
        solar_col = st.selectbox("일사량", sensor_df.columns)

    st.markdown("---")

    # ✅ 수확량 컬럼 선택 - 가로 배치
    st.markdown("**수확량 데이터 컬럼 선택**")
    col6, col7, col8 = st.columns(3)

    with col6:
        date_col_yield = st.selectbox("조사일자", yield_df.columns)
    with col7:
        harvest_count_col = st.selectbox("수확수", yield_df.columns)
    with col8:
        harvest_weight_col = st.selectbox("평균과중", yield_df.columns)

    st.markdown("---")

    # ✅ 생육 컬럼 선택 - 가로 3개씩 여러 줄 배치
    st.markdown("**추가 생육 컬럼 선택**")
    growth_features = ["초장", "생장길이", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"]
    growth_cols = {}

    # 3개씩 끊어서 가로로 배치
    for i in range(0, len(growth_features), 3):
        cols = st.columns(3)
        for j, gf in enumerate(growth_features[i:i + 3]):
            with cols[j]:
                if gf in yield_df.columns:
                    growth_cols[gf] = st.selectbox(
                        f"{gf}",
                        [None] + yield_df.columns.tolist(),
                        index=yield_df.columns.get_loc(gf) + 1
                    )
                else:
                    growth_cols[gf] = st.selectbox(
                        f"{gf}",
                        [None] + yield_df.columns.tolist(),
                        index=0
                    )

    # --- 날짜 변환 ---
    sensor_df[date_col_sensor] = pd.to_datetime(sensor_df[date_col_sensor])
    yield_df[date_col_yield] = pd.to_datetime(yield_df[date_col_yield])

    sensor_df['date'] = sensor_df[date_col_sensor].dt.date
    sensor_df['hour'] = sensor_df[date_col_sensor].dt.hour
    sensor_df['time'] = sensor_df[date_col_sensor].dt.time

    # --- 주 선택 슬라이더 동기화 ---
    if "weeks" not in st.session_state:
        st.session_state.weeks = 7  # 초기값


    def update_weeks_1():
        st.session_state.weeks = st.session_state.weeks_slider_1


    def update_weeks_2():
        st.session_state.weeks = st.session_state.weeks_slider_2


    weeks1 = st.slider("평균 계산 기간 (주 단위) - 센서 평균용",
                       1, 7, st.session_state.weeks, key="weeks_slider_1", on_change=update_weeks_1)
    days = st.session_state.weeks * 7

    # --- 평균 계산 ---
    results = []
    for idx, row in yield_df.iterrows():
        date = row[date_col_yield]
        start_date = date - timedelta(days=days)
        mask = (sensor_df[date_col_sensor] >= start_date) & (sensor_df[date_col_sensor] <= date)
        subset = sensor_df.loc[mask]
        if not subset.empty:
            # 일사량 0시 기준
            midnight_values = subset[subset["time"].astype(str) == "00:00:00"]
            midnight_daily = midnight_values.groupby("date")[solar_col].first().reset_index()
            avg_solar = midnight_daily[solar_col].mean() if not midnight_daily.empty else None

            # CO2 06~18시
            co2_daytime = subset[(subset["hour"] >= 6) & (subset["hour"] <= 18)]
            co2_daily_mean = co2_daytime.groupby("date")[co2_col].mean().reset_index()
            avg_co2 = co2_daily_mean[co2_col].mean() if not co2_daily_mean.empty else None

            # 온도/습도 24시간 평균
            avg_temp = subset[temp_col].mean()
            avg_hum = subset[hum_col].mean()

            result_row = {
                "조사일자": date,
                "수확수": row[harvest_count_col],
                "평균과중": row[harvest_weight_col],
                "평균온도": avg_temp,
                "평균습도": avg_hum,
                f"{days}일평균CO₂(06~18시)": avg_co2,
                f"{days}일평균누적일사량(0:00기준)": avg_solar
            }

            for gf, col in growth_cols.items():
                result_row[gf] = row[col] if col is not None else None

            results.append(result_row)

    df = pd.DataFrame(results)
    st.subheader("매핑 데이터")
    st.dataframe(df)

    # 환경 컬럼 매핑 (df의 컬럼 이름 기준)
    env_mapping = {
        "평균온도": "평균온도",
        "평균습도": "평균습도",
        f"{days}일평균CO₂(06~18시)": f"{days}일평균CO₂(06~18시)",
        f"{days}일평균누적일사량(0:00기준)": f"{days}일평균누적일사량(0:00기준)"
    }

    env_cols = st.multiselect(
        "환경 그래프로 표시할 항목 선택",
        list(env_mapping.keys()),
        default=list(env_mapping.keys())  # 기본으로 4개 다 선택
    )

    if env_cols:
        # 2행 2열 배치로 시계열 그래프 출력
        for i in range(0, len(env_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(env_cols[i:i + 2]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(df["조사일자"], df[env_mapping[col_name]], marker="o", linestyle="-")
                    ax.set_title(f"{col_name} 시계열")
                    ax.set_xlabel("조사일자")
                    ax.set_ylabel(col_name)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig)
                    plt.close(fig)

    # --- 📈 조사일자별 시계열 그래프 ---
    st.subheader("📈 조사일자 기준 시계열 그래프")

    # 날짜 정렬
    df = df.sort_values("조사일자")

    # 그래프 대상 컬럼 선택 (수확수~화방높이)
    plot_cols = st.multiselect(
        "그래프로 표시할 항목 선택",
        ["수확수", "평균과중"] + growth_features,
        default=["수확수", "평균과중"]
    )

    if plot_cols:
        # 3개씩 가로로 그래프 배치
        for i in range(0, len(plot_cols), 3):
            cols = st.columns(3)
            for j, col_name in enumerate(plot_cols[i:i + 3]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4.5, 3))
                    ax.plot(df["조사일자"], df[col_name], marker="o", linestyle="-")
                    ax.set_title(f"{col_name} 시계열")
                    ax.set_xlabel("조사일자")
                    ax.set_ylabel(col_name)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig)
                    plt.close(fig)

    # --- 🌿 환경 vs 생육 2축 시계열 그래프 (4개 비교, 숨기기 기능) ---
    st.subheader("🌿 환경 vs 생육 2축 시계열 그래프 (4개 비교)")

    # 환경 변수 목록
    env_options = [
        "평균온도",
        "평균습도",
        f"{days}일평균CO₂(06~18시)",
        f"{days}일평균누적일사량(0:00기준)"
    ]

    # 생육/수확 변수 목록
    growth_options = ["수확수", "평균과중", "초장", "엽수", "엽장", "엽폭", "생장길이", "줄기굵기", "화방높이"]

    # 3개의 컬럼(그래프) 배치
    cols = st.columns(4)

    for i in range(4):
        with cols[i]:
            st.markdown(f"#### 그래프 {i + 1}")

            # 체크박스로 그래프 숨기기 기능
            show_graph = st.checkbox(f"그래프 {i + 1} 표시", value=True, key=f"show_{i}")

            if show_graph:
                # 환경 / 생육 변수 선택
                selected_env = st.selectbox(f"환경 변수 {i + 1}", env_options, key=f"env_{i}")
                selected_growth = st.selectbox(f"생육/수확 변수 {i + 1}", growth_options, index=0, key=f"growth_{i}")

                # 그래프 그리기
                if selected_env and selected_growth:
                    fig, ax1 = plt.subplots(figsize=(5, 4))

                    # 왼쪽 y축: 환경
                    color1 = "tab:blue"
                    ax1.set_xlabel("조사일자")
                    ax1.set_ylabel(selected_env, color=color1)
                    ax1.plot(df["조사일자"], df[selected_env], color=color1, marker="o", label=selected_env)
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, linestyle="--", alpha=0.4)

                    # 오른쪽 y축: 생육/수확
                    ax2 = ax1.twinx()
                    color2 = "tab:red"
                    ax2.set_ylabel(selected_growth, color=color2)
                    ax2.plot(df["조사일자"], df[selected_growth], color=color2, marker="s", linestyle="--",
                             label=selected_growth)
                    ax2.tick_params(axis='y', labelcolor=color2)

                    # 범례
                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

                    st.pyplot(fig)
                    plt.close(fig)

    # --- 🌿 환경요소 vs 생육컬럼 2축 시계열 그래프 (Matplotlib 2×2) ---
    st.subheader("🌿 환경요소 vs 생육컬럼 2축 시계열 그래프 (자동 4개 조합)")

    # 생육 컬럼 선택 (수확수 ~ 화방높이)
    growth_choice = st.selectbox(
        "생육 컬럼 선택 (2축 그래프에서 표시할 항목)",
        ["수확수", "평균과중"] + growth_features,
        index=0
    )

    env_list = [
        ("평균온도", "평균온도"),
        ("평균습도", "평균습도"),
        ("평균CO₂", f"{days}일평균CO₂(06~18시)"),
        ("평균누적일사량", f"{days}일평균누적일사량(0:00기준)")
    ]

    # 2×2 레이아웃
    for i in range(0, len(env_list), 2):
        cols = st.columns(2)
        for j, (title, col_name) in enumerate(env_list[i:i + 2]):
            with cols[j]:
                fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

                # 환경 (왼쪽 y축)
                color1 = "tab:blue"
                ax1.set_xlabel("조사일자")
                ax1.set_ylabel(title, color=color1)
                ax1.plot(df["조사일자"], df[col_name], color=color1, marker="o", label=title)
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, linestyle="--", alpha=0.4)

                # 생육 (오른쪽 y축)
                ax2 = ax1.twinx()
                color2 = "tab:red"
                ax2.set_ylabel(growth_choice, color=color2)
                ax2.plot(df["조사일자"], df[growth_choice], color=color2, marker="s", linestyle="--", label=growth_choice)
                ax2.tick_params(axis='y', labelcolor=color2)

                # 범례
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8)

                ax1.set_title(f"{title} vs {growth_choice}", fontsize=11)
                st.pyplot(fig)
                plt.close(fig)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("🌿 환경요소 vs 생육컬럼 2축 시계열 그래프 (Plotly 인터랙티브 2×2)")

    # 생육 컬럼 선택 (수확수~화방높이)
    growth_choice_plotly = st.selectbox(
        "생육 컬럼 선택 (Plotly 그래프용)",
        ["수확수", "평균과중"] + growth_features,
        index=0,
        key="plotly_growth_choice"
    )

    env_list = [
        ("평균온도", "평균온도"),
        ("평균습도", "평균습도"),
        ("평균CO₂", f"{days}일평균CO₂(06~18시)"),
        ("평균누적일사량", f"{days}일평균누적일사량(0:00기준)")
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{title} vs {growth_choice_plotly}" for title, _ in env_list],
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )

    for idx, (title, env_col) in enumerate(env_list):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # 환경 (왼쪽 y축)
        fig.add_trace(
            go.Scatter(
                x=df["조사일자"],
                y=df[env_col],
                mode='lines+markers',
                name=title,
                line=dict(color='blue'),
                hovertemplate=f"{title}: %{{y}}<br>조사일자: %{{x}}"
            ),
            row=row, col=col, secondary_y=False
        )

        # 생육 (오른쪽 y축)
        fig.add_trace(
            go.Scatter(
                x=df["조사일자"],
                y=df[growth_choice_plotly],
                mode='lines+markers',
                name=growth_choice_plotly,
                line=dict(color='red', dash='dash'),
                hovertemplate=f"{growth_choice_plotly}: %{{y}}<br>조사일자: %{{x}}"
            ),
            row=row, col=col, secondary_y=True
        )

        fig.update_yaxes(title_text=title, row=row, col=col, secondary_y=False)
        fig.update_yaxes(title_text=growth_choice_plotly, row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=800,
        width=950,
        title_text="환경요소 vs 생육컬럼 2축 시계열 (인터랙티브)",
        showlegend=True,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=60, b=30)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 모델 선택 ---
    st.subheader("모델 선택")
    model_options = ["RandomForest", "GradientBoosting", "XGBoost", "LGBM", "GaussianNB"]
    model_choice = st.selectbox("모델 선택", model_options)

    target_col = st.selectbox("예측 대상 컬럼 선택", ["수확수", "평균과중"] + growth_features)
    features = [col for col in df.columns if col not in ["조사일자", "수확수", "평균과중"] + growth_features]

    X = df[features]
    y = df[target_col]
    X = X.fillna(X.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "RandomForest":
        model = RandomForestRegressor(random_state=42)
    elif model_choice == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_choice == "XGBoost":
        model = XGBRegressor(random_state=42)
    elif model_choice == "LGBM":
        model = LGBMRegressor(random_state=42)
    elif model_choice == "GaussianNB":
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    weeks2 = st.slider("평균 계산 기간 (주 단위) - 모델용",
                       1, 7, st.session_state.weeks, key="weeks_slider_2", on_change=update_weeks_2)
    days = st.session_state.weeks * 7

    # --- 평가지표 ---
    st.subheader("모델 평가 지표")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    st.write(f"R²: {r2_score(y_test, y_pred):.3f}")

    # ---------------------------
    # SHAP, Feature Importance 레이아웃 재배치 및 ICE/PDP/ALE 추가
    # ---------------------------

    import math
    from sklearn.utils import check_array


    # 간단한 ALE 계산 함수 (수치형 feature 전용, 모델의 predict 사용)
    def compute_ale(model, X, feature, bins=10):
        """
        간단한 1차원 ALE 근사
        model: 학습된 모델 (predict 메서드 사용)
        X: DataFrame (원본 특성 행렬)
        feature: feature 이름(string)
        bins: bin 수
        returns: bin_centers, ale_values
        """
        x = X[feature].values
        # remove nan rows for feature
        mask = ~np.isnan(x)
        x = x[mask]
        X_valid = X.loc[mask].reset_index(drop=True)
        percentiles = np.linspace(0, 100, bins + 1)
        cutpoints = np.percentile(x, percentiles)
        # 중복 컷포인트 처리: 유니크로
        cutpoints = np.unique(cutpoints)
        if len(cutpoints) < 2:
            # 변동이 거의 없을 때
            return np.array([np.mean(x)]), np.array([0.0])

        # 각 구간별 평균 기여 계산
        local_effects = []
        bin_centers = []
        for i in range(len(cutpoints) - 1):
            lo, hi = cutpoints[i], cutpoints[i + 1]
            # 해당 구간에 속하는 인덱스
            in_bin = (X_valid[feature] >= lo) & (X_valid[feature] <= hi)
            if in_bin.sum() == 0:
                # 해당 구간에 점이 없으면 0 넣기
                local_effects.append(0.0)
                bin_centers.append((lo + hi) / 2.0)
                continue
            X_lo = X_valid.copy()
            X_hi = X_valid.copy()
            # 왼쪽 경계값으로, 오른쪽 경계값으로 바꿔서 예측 차이를 봄
            X_lo.loc[in_bin, feature] = lo
            X_hi.loc[in_bin, feature] = hi
            try:
                preds_hi = model.predict(X_hi)
                preds_lo = model.predict(X_lo)
            except Exception:
                # some models require numpy array
                preds_hi = model.predict(X_hi.values)
                preds_lo = model.predict(X_lo.values)
            diff = preds_hi - preds_lo
            # 지역 평균 기여
            local_effect = diff[in_bin.values].mean() if in_bin.sum() > 0 else 0.0
            local_effects.append(local_effect)
            bin_centers.append((lo + hi) / 2.0)

        # 누적합으로 ALE 계산 (baseline을 0으로 맞춤)
        ale = np.cumsum(local_effects)
        # 평균을 0 기준으로 조정
        ale = ale - ale.mean()
        return np.array(bin_centers), ale


    # ---------- SHAP + Feature Importance 레이아웃 ----------
    st.subheader("SHAP 해석 및 Feature Importance")

    # 상단: 두 컬럼으로 배치 (왼쪽 SHAP plot, 오른쪽 Feature Importance plot)
    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        st.markdown("### 🔍 SHAP Summary (샘플 중요도 시각화)")
        if model_choice != "GaussianNB":
            try:
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                # SHAP summary plot (matplotlib)
                fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
                # summary_plot은 내부에서 figure를 생성하므로 show=False 옵션 사용
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig_shap)
                plt.close(fig_shap)
            except Exception as e:
                st.error(f"SHAP 시각화 중 오류 발생: {e}")
        else:
            st.info("GaussianNB 모델은 SHAP 해석을 지원하지 않습니다.")

    with top_col2:
        st.markdown("### 📊 Feature Importance (모델 기반)")
        try:
            # 모델에 feature_importances_가 있는 경우
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
                    by="Importance", ascending=False
                )
            else:
                # 없는 경우(예: GaussianNB), 간단한 대체: permutation importance를 권장하지만 여기서는 계층적 대체
                fi_df = pd.DataFrame({"Feature": features, "Importance": np.zeros(len(features))})
                st.warning("선택한 모델에 feature_importances_ 속성이 없습니다. 중요도는 0으로 표시됩니다.")
            # 막대그래프
            fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
            ax_fi.barh(fi_df["Feature"], fi_df["Importance"])
            ax_fi.invert_yaxis()
            ax_fi.set_title("Feature Importance")
            st.pyplot(fig_fi)
            plt.close(fig_fi)
        except Exception as e:
            st.error(f"Feature Importance 생성 중 오류: {e}")

    # 하단: shap_summary 표(왼쪽) 및 Feature Importance 표(오른쪽)
    bot_col1, bot_col2 = st.columns([1, 1])
    with bot_col1:
        st.markdown("#### SHAP 영향력 요약 (Mean |SHAP|)")
        if model_choice != "GaussianNB":
            try:
                shap_mean = np.abs(shap_values.values).mean(axis=0)
                shap_summary = pd.DataFrame({"Feature": features, "Mean(|SHAP value|)": shap_mean}) \
                    .sort_values(by="Mean(|SHAP value|)", ascending=False)
                st.dataframe(shap_summary)
            except Exception as e:
                st.error(f"SHAP 요약표 생성 오류: {e}")
        else:
            st.info("GaussianNB 모델은 SHAP 해석을 지원하지 않습니다.")

    with bot_col2:
        st.markdown("#### Feature Importance Table")
        try:
            st.dataframe(fi_df.reset_index(drop=True))
        except Exception as e:
            st.error(f"Feature Importance 표 출력 오류: {e}")

    # ---------- ICE, PDP, ALE 세 그래프 나란히 배치 ----------
    st.subheader("ICE vs PDP vs ALE (선택한 Feature 기준)")

    # 분석할 feature 선택 (앞과 동일하게 features 사용)
    ice_feature = st.selectbox("분석할 Feature 선택 (ICE/PDP/ALE)", features, key="ice_feature_global")
    n_samples = st.slider("ICE 샘플 수 (최대)", 1, max(1, len(X_test)), value=min(50, len(X_test)), key="ice_samples_global")
    ale_bins = st.slider("ALE bins 수", 4, 30, 10)

    # 컬럼 배치: 3개 나란히
    col_ice, col_pdp, col_ale = st.columns(3)

    # ICE plot
    with col_ice:
        st.markdown("**ICE Plot**")
        try:
            fig_ice, ax_ice = plt.subplots(figsize=(5, 3))
            # 개별 곡선: sample n_samples
            # PartialDependenceDisplay can draw individuals
            try:
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_test.sample(n=n_samples, random_state=42),
                    features=[ice_feature],
                    kind="individual",
                    ax=ax_ice,
                    line_kw={"alpha": 0.3},
                )
            except Exception:
                # sklearn 버전에 따라 API 다를 수 있음 - 예외 시 직접 그리기 시도(간단)
                Xs = X_test.sample(n=n_samples, random_state=42)
                xs = np.linspace(Xs[ice_feature].min(), Xs[ice_feature].max(), 50)
                for _, row in Xs.iterrows():
                    Xtmp = pd.DataFrame(np.tile(row.values, (len(xs), 1)), columns=Xs.columns)
                    Xtmp[ice_feature] = xs
                    preds = model.predict(Xtmp)
                    ax_ice.plot(xs, preds, alpha=0.2)
            ax_ice.set_title(f"ICE: {ice_feature}")
            ax_ice.set_xlabel(ice_feature)
            ax_ice.set_ylabel("Predicted")
            st.pyplot(fig_ice)
            plt.close(fig_ice)
        except Exception as e:
            st.error(f"ICE 시각화 오류: {e}")

    # PDP plot
    with col_pdp:
        st.markdown("**PDP (Partial Dependence)**")
        try:
            fig_pdp, ax_pdp = plt.subplots(figsize=(5, 3))
            PartialDependenceDisplay.from_estimator(
                model,
                X_test,
                features=[ice_feature],
                kind="average",
                ax=ax_pdp,
                line_kw={"color": "red"},
            )
            ax_pdp.set_title(f"PDP: {ice_feature}")
            ax_pdp.set_xlabel(ice_feature)
            ax_pdp.set_ylabel("Predicted")
            st.pyplot(fig_pdp)
            plt.close(fig_pdp)
        except Exception as e:
            st.error(f"PDP 시각화 오류: {e}")

    # ALE plot
    with col_ale:
        st.markdown("**ALE (Approx.)**")
        try:
            bin_centers, ale_vals = compute_ale(model, X_test.reset_index(drop=True), ice_feature, bins=ale_bins)
            fig_ale, ax_ale = plt.subplots(figsize=(5, 3))
            if len(bin_centers) == 1:
                ax_ale.hlines(0, bin_centers[0] - 0.5, bin_centers[0] + 0.5)
            else:
                ax_ale.plot(bin_centers, ale_vals, marker="o", linestyle="-")
            ax_ale.set_title(f"ALE (approx): {ice_feature}")
            ax_ale.set_xlabel(ice_feature)
            ax_ale.set_ylabel("ALE")
            st.pyplot(fig_ale)
            plt.close(fig_ale)
        except Exception as e:
            st.error(f"ALE 계산/시각화 오류: {e}")


