import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

#폰트설정(한글)
KOREAN_FONT = dict(family="Malgun Gothic")

@st.cache_data
def load_data():
        df1 = pd.read_csv("./1_team/res/소비자물가지수.csv", encoding='utf-8-sig')
        df2 = pd.read_csv("./1_team/res/유가.csv", encoding='utf-8-sig')
        df3 = pd.read_csv("./1_team/res/환율.csv", encoding='utf-8-sig')
        df4 = pd.read_csv("./1_team/res/통합_2.csv", encoding='utf-8-sig')
        return df1, df2, df3, df4

X_data = np.array([
    
    # 환율 데이터
    [1.297, 0.691, 0.17, 0.177, 0.689, 1.146, 1.43, 0.793, 1.135, 0.008, 0.067, 0.871, 0.363, 1.893, 0.926, 1.21, 1.814, 0.068, 0.52, 0.701, 0.771, 0.038, 0.561, 0.134, 0.128, 0.229, 0.614, 0.002, 0.117, 0.27, 0.351, 0.095, 0.093, 0.078, 0.231, 1.189, 0.256, 0.506, 0.228, 0.095, 0.114, 0.064, 0.54, 0.357, 0.235, 0.454],

    # 본원 통화
    [3.177, 0.477, -0.388, 1.69, 2.429, -0.291, 0.876, 1.282, 2.695, -0.517, 1.079, 2.175, -0.155, 0.155, 1.78, 0.342, 0.038, 3.258, -2.605, 0.565, 1.536, -3.615, 0.306, -0.534, -0.115, -1.459, 0.273, 2.021, -0.838, 1.729, -0.944, 0.076, 0.305, -1.937, 1.472, 0.802, 1.515, 0.261, -0.707, 1.199, -0.222, 1.633, -0.475, 0.807, 0.182, -0.509],

    # 유가(휘발유)
    [1.483, 3.421, 1.404, 0.456, 2.324, 3.292, 1.012, -0.187, 4.239, 1.464, -5.239, -0.677, 4.855, 13.055, 1.964, -0.479, 5.944, -2.592, -11.714, -3.47, -3.662, -0.98, -5.25, -0.048, 0.996, 0.872, 3.059, -0.74, -2.957, 0.306, 8.28, 3.052, 0.381, -5.171, -4.957, -1.957, 2.884, 1.523, 2.971, 0.574, -2.362, 3.002, -0.928, -4.084, -1.902, 2.322],

    # 유가(경유)
    [1.691, 3.9, 1.532, 0.455, 2.655, 3.724, 1.052, -0.234, 5.018, 2.679, -5.215, -1.046, 5.718, 18.891, 4.351, 3.035, 6.351, -0.197, -9.382, -2.07, -0.641, 2.22, -5.105, -6.048, -4.116, -4.151, -0.261, -4.15, -5.264, 0.143, 12.652, 5.935, 1.427, -3.673, -6.259, -3.03, 2.546, 1.398, 1.222, -1.167, -3.381, 3.692, -0.882, -4.62, -2.522, 2.786],
    ]
)

Y_target = np.array([0.534, 0.256, 0.137, 0.069, 0.0, 0.206, 0.479, 0.409, 0.174, 0.503, 0.164, 0.779, 0.544, 0.645, 0.688, 0.627, 0.66, 0.481, -0.092, 0.175, 0.312, -0.082, 0.174, 0.741, 0.236, 0.172, 0.226, 0.325, 0.027, 0.117, 0.89, 0.508, 0.372, -0.521, 0.044, 0.39, 0.539, 0.149, 0.053, 0.079, -0.228, 0.255, 0.359, 0.096, 0.035, -0.253])

# model = LinearRegression()
# model.fit(X_data.T, Y_target)

b0 = 0.17835904841144312
weights = np.array([0.11770982, 0.04759305, 0.0220467, 0.00542832])
y_pred = b0 + np.dot(X_data.T, weights)

#3. Streamlit 
st.set_page_config(page_title='물가지수 예측 프로그램', layout="wide")

with st.sidebar:
    # 이미지 삽입하기
    st.image("https://cdn-icons-png.flaticon.com/512/5133/5133850.png", width=200)

    name = st.text_input("이름을 입력하세요", value="사용자")
    st.write(f"안녕하세요 {name}님!!")

    st.divider()
    with st.expander("ℹ️ 데이터 출처"):
       st.caption("""

        - 소비자물가지수: kosis국가통계 포털

        - 환율/유가: 공공 데이터포털

        - 기준년도: 2021~2025

        """)

tab1, tab2, tab3 = st.tabs(['🔮 미래 예측', '📂 데이터 보기', '📊 시각화 분석'])

#tab 1: 예측 화면
with tab1:
    st.markdown("---")
    st.write("### 🔮 미래 가격 상승률 예측하기")
    st.info("경제 지표 변화율을 직접 입력하고 내년 물가 상승률을 예측해보기.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val1 = st.number_input("환율 변화율 (%)", value=1.5)
    with c2:
        val2 = st.number_input("본원통화 변화율 (%)", value=0.5)
    with c3:
        val3 = st.number_input("휘발유 변화율 (%)", value=-1.0)
    with c4:
        val4 = st.number_input("경유 변화율 (%)", value=-1.0)

    user_input = np.array([val1, val2, val3, val4])

    # 예측 계산
    result = b0 + np.dot(user_input, weights)

    st.divider()
    st.metric(label="내년 예상 물가 상승률", value=f"{result:.3f} %", delta=f"{result - 2.2:.3f} % (전년비)")
    if result > 2.0:
        st.warning(f"⚠️ 예측 결과, 물가가 약 {result:.2f}% 상승할 것으로 보입니다.")
    else:
        st.success(f"✅ 예측 결과, 물가 상승률이 {result:.2f}%로 비교적 안정적일 것으로 보입니다.")

#tab 2: 데이터 보기 
with tab2:
    st.subheader("사용한 데이터 자료 보기")
    df1, df2, df3, df4 = load_data() 
    select_data = st.selectbox(" ", ['물가 데이터', '유가 데이터', '환율 데이터', '통합 데이터'])
    
    if select_data == '물가 데이터':
        st.dataframe(df1, use_container_width=True)

    elif select_data == '유가 데이터':
        st.dataframe(df2, use_container_width=True)

    elif select_data == '환율 데이터':
        st.dataframe(df3, use_container_width=True)

    else:
        st.dataframe(df4, use_container_width=True)

# tab 3: 시각화 화면
with tab3:
    st.header("📊 분석 결과 시각화")
    
    # 날짜 및 예측값 데이터 생성
    dt_range = pd.date_range(start="2021-01-01", periods=len(Y_target), freq='M')
    # y_pred = model.predict(X_data.T)
    
    # 1. 산점도 그래프
    with st.expander("1. 과거 변화율 분포"):
        f1 = px.scatter(x = range(len(Y_target)), y = Y_target, title = "과거 변화율 분포", labels = {"x": "인덱스(시간)", "y": "변화율(%)"})
        f1.add_hline(y=0, line_dash="dash", line_color="red")

        f1.update_layout(font=KOREAN_FONT)
        st.plotly_chart(f1, use_container_width=True)
        
        st.info("5년간 물가 변동 데이터를 점으로 찍은 그래프, 과거 전반의 변동 폭 가늠할 수 있음")

    # 2. 실제값과 예측값 비교 선그래프
    with st.expander("2. 실제값과 예측값 비교 선그래프"):
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=dt_range, y=Y_target, name="실제값", line=dict(color="blue")))
        f2.add_trace(go.Scatter(x=dt_range, y= y_pred, name="예측값", line=dict(color="red", dash="dash")))
        
        f2.update_layout(title="실제값과 예측값 추이 비교", font=KOREAN_FONT)
        st.plotly_chart(f2, use_container_width=True)

        st.info("과거 2021~2023의 경제 상황은 잘 설명하고 있으나 이후 차이폭이 커짐 이는 외부 변수의 영향력이 더 커졌거나 예외적 경제 충격의 영향으로 볼 수 있음" )

    # 3. 주요 지표 간 상관관계 분석(히트맵)
    with st.expander("4. 경제 지표 간 상관관계 분석(Heatmap)"):
        corr_data = pd.DataFrame(X_data.T, columns = ['환율','본원통화','휘발유','경유'])
        corr_data['물가지수'] = Y_target

        # 상관계수 계산하기
        df_corr = corr_data.corr()

        # 히트맵 
        f4 = px.imshow(
            df_corr,
            text_auto='.3f',
            title = "경제 지표 및 물가 상관관계 히트맵"
        )
        f4.update_layout(font = KOREAN_FONT, xaxis_title = "경제 지표", yaxis_title = "경제 지표")
        st.plotly_chart(f4, use_container_width=True)

