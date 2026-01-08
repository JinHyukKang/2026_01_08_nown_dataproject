import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. 페이지 설정 및 세션 상태 초기화
# ---------------------------------------------------------
st.set_page_config(
    page_title="비만 등급 예측 AI",
    page_icon="🩺",
    layout="wide"
)

if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'input_df' not in st.session_state:
    st.session_state['input_df'] = None

# ---------------------------------------------------------
# 2. 모델 불러오기
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load('./3_team/obesity_model.pkl')
    except:
        st.error("❌ 모델 파일(obesity_model.pkl)을 찾을 수 없습니다.")
        return None

model = load_model()

# ---------------------------------------------------------
# 3. 화면 구성
# ---------------------------------------------------------
st.title("🩺 비만 등급 예측 솔루션")
st.markdown("---")

tab1, tab2 = st.tabs(["📝 정보 입력 (Input)", "📊 진단 결과 (Result)"])

# =========================================================
# [탭 1] 정보 입력 페이지
# =========================================================
with tab1:
    st.header("사용자 정보를 입력해주세요")
    st.info("정확한 분석을 위해 솔직하게 선택해주세요.")

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # 1. 개인 신상 & 이동수단
    # -----------------------------------------------------
    with col1:
        st.subheader("1. 기본 정보")
        gender = st.radio("성별", ["남성", "여성"], horizontal=True)
        age = st.number_input("나이 (만)", 10, 100, 25)
        family_history = st.radio("가족 비만력 (부모/형제)", ["예", "아니오"], horizontal=True)
        
        st.divider()
        st.subheader("4. 핵심 정보 (이동수단)")
        mtrans_option = st.selectbox("주 이용 교통수단", ["자동차", "오토바이", "자전거", "대중교통", "도보"])

    # -----------------------------------------------------
    # 2. 식습관
    # -----------------------------------------------------
    with col2:
        st.subheader("2. 식습관")
        
        # FAVC
        st.write("**고칼로리 음식 섭취**")
        st.caption("※ 튀김, 패스트푸드, 디저트 등을 자주 드시나요?")
        favc = st.radio("고칼로리 섭취 여부", ["예", "아니오"], horizontal=True, label_visibility="collapsed")
        
        # FCVC (채소)
        fcvc_label = st.selectbox("채소 섭취 빈도", 
                                ["거의 안 먹음", "가끔 먹음", "매끼 먹음"])
        fcvc_map = {"거의 안 먹음": 1.0, "가끔 먹음": 2.0, "매끼 먹음": 3.0}
        
        # NCP (식사 횟수)
        ncp_label = st.selectbox("하루 식사 횟수", 
                               ["1끼", "2끼", "3끼", "4끼 이상"])
        ncp_map = {"1끼": 1.0, "2끼": 2.0, "3끼": 3.0, "4끼 이상": 4.0}
        
        # [수정] CAEC (간식) - 구체적 횟수 명시
        caec_label = st.selectbox("식사 외 간식 섭취", 
                                ["안 먹음", 
                                 "가끔 (주 1~2회)", 
                                 "자주 (주 3~4회)", 
                                 "항상 (매일)"])
        caec_map = {
            "안 먹음": 0,
            "가끔 (주 1~2회)": 1,
            "자주 (주 3~4회)": 2,
            "항상 (매일)": 3
        }

        smoke = st.radio("흡연 여부", ["예", "아니오"], horizontal=True)

    # -----------------------------------------------------
    # 3. 생활 습관
    # -----------------------------------------------------
    with col3:
        st.subheader("3. 생활 습관")
        
        # CH2O (물)
        ch2o_label = st.selectbox("하루 물 섭취량", 
                                ["1L 미만 (거의 안 마심)", "1L ~ 2L (보통)", "2L 이상 (많이 마심)"])
        ch2o_map = {"1L 미만 (거의 안 마심)": 1.0, "1L ~ 2L (보통)": 2.0, "2L 이상 (많이 마심)": 3.0}
        
        # [수정] CALC (음주) - 요청하신 기준 적용
        calc_label = st.selectbox("음주 빈도", 
                                ["마시지 않음", 
                                 "가끔 마심 (주 1~2회)", 
                                 "자주 마심 (주 3~4회)", 
                                 "항상 마심 (주 5회 이상)"])
        calc_map = {
            "마시지 않음": 0,
            "가끔 마심 (주 1~2회)": 1,
            "자주 마심 (주 3~4회)": 2,
            "항상 마심 (주 5회 이상)": 3
        }

        scc = st.radio("칼로리 계산(다이어트) 여부", ["예", "아니오"], horizontal=True)
        
        # FAF (운동)
        faf_label = st.selectbox("일주일 운동 빈도", 
                               ["운동 안 함", "주 1~2일", "주 3~4일", "주 5일 이상"])
        faf_map = {"운동 안 함": 0.0, "주 1~2일": 1.0, "주 3~4일": 2.0, "주 5일 이상": 3.0}
        
        # TUE (전자기기)
        tue_label = st.selectbox("하루 전자기기 사용 (스마트폰/PC)", 
                               ["0~2시간 (적음)", "3~5시간 (보통)", "5시간 이상 (많음)"])
        tue_map = {"0~2시간 (적음)": 0.0, "3~5시간 (보통)": 1.0, "5시간 이상 (많음)": 2.0}

    # -----------------------------------------------------
    # 진단 버튼
    # -----------------------------------------------------
    st.markdown("---")
    _, center_col, _ = st.columns([1, 1, 1])
    
    with center_col:
        btn_click = st.button("🚀 진단 시작하기", type="primary", use_container_width=True)

    if btn_click:
        # 입력값 변환
        input_data = {
            'Gender': 1 if gender == "남성" else 0,
            'Age': age,
            'family_history_with_overweight': 1 if family_history == "예" else 0,
            'FAVC': 1 if favc == "예" else 0,
            'FCVC': fcvc_map[fcvc_label],
            'NCP': ncp_map[ncp_label],
            'CAEC': caec_map[caec_label],  # 간식 매핑 적용
            'SMOKE': 1 if smoke == "예" else 0,
            'CH2O': ch2o_map[ch2o_label],
            'SCC': 1 if scc == "예" else 0,
            'FAF': faf_map[faf_label],
            'TUE': tue_map[tue_label],
            'CALC': calc_map[calc_label],  # 음주 매핑 적용
            
            # 이동수단 One-Hot Encoding
            'MTRANS_Automobile': 1 if mtrans_option == "자동차" else 0,
            'MTRANS_Bike': 1 if mtrans_option == "자전거" else 0,
            'MTRANS_Motorbike': 1 if mtrans_option == "오토바이" else 0,
            'MTRANS_Public_Transportation': 1 if mtrans_option == "대중교통" else 0,
            'MTRANS_Walking': 1 if mtrans_option == "도보" else 0
        }
        
        # DataFrame 생성 및 세션 저장
        input_df = pd.DataFrame([input_data])
        st.session_state['input_df'] = input_df
        
        if model is not None:
            pred = model.predict(input_df)[0]
            st.session_state['prediction_result'] = pred
            st.success("✅ 진단 완료! 상단 [진단 결과] 탭을 확인하세요.")
            st.rerun()
        else:
            st.error("모델 로드 실패")

# =========================================================
# [탭 2] 진단 결과 페이지
# =========================================================
with tab2:
    st.header("📊 AI 진단 결과 리포트")
    
    if st.session_state['prediction_result'] is None:
        st.warning("👈 [정보 입력] 탭에서 데이터를 입력하고 진단 버튼을 눌러주세요.")
    
    else:
        pred = st.session_state['prediction_result']
        
        grades = {
            0: "저체중 (Insufficient Weight)",
            1: "정상 체중 (Normal Weight)",
            2: "과체중 (Overweight)",
            3: "비만 (Obesity Type I, II)",
            4: "고도비만 (Obesity Type III)"
        }
        result_text = grades.get(pred, "알 수 없음")

        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="당신의 비만 등급", value=f"Level {pred}")
            
            if pred == 0: st.image("https://cdn-icons-png.flaticon.com/512/3048/3048384.png", width=150)
            elif pred == 1: st.image("https://cdn-icons-png.flaticon.com/512/4359/4359853.png", width=150)
            elif pred == 2: st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=150)
            else: st.image("https://cdn-icons-png.flaticon.com/512/3076/3076899.png", width=150)

        with col_res2:
            if pred == 0:
                st.info(f"### 결과: {result_text}")
                st.write("체중이 평균보다 적게 나갑니다. 규칙적인 식사와 근력 운동을 권장합니다.")
            elif pred == 1:
                st.success(f"### 결과: {result_text}")
                st.write("축하합니다! 현재 아주 건강한 상태입니다. 지금의 습관을 유지하세요.")
            elif pred == 2:
                st.warning(f"### 결과: {result_text}")
                st.write("체중 관리가 필요한 '주의' 단계입니다. 운동량을 조금 더 늘려보세요.")
            else:
                st.error(f"### 결과: {result_text}")
                st.write("적극적인 체중 감량이 필요합니다. 전문가의 도움을 받거나 생활 습관을 크게 개선해야 합니다.")

        st.divider()
        st.subheader("💡 AI 맞춤 솔루션")
        
        advice_list = []
        user_data = st.session_state['input_df'].iloc[0]
        
        # 상세 조언 로직
        if user_data['TUE'] >= 2.0:
            advice_list.append("- **스마트폰/PC 사용**이 많습니다. 앉아있는 시간을 줄여보세요.")
        if user_data['FCVC'] < 2.0:
            advice_list.append("- **채소 섭취**가 부족합니다. 매끼 식단에 채소를 추가하세요.")
        if user_data['MTRANS_Automobile'] == 1:
            advice_list.append("- **자가용** 대신 대중교통이나 걷기를 생활화 해보세요.")
        if user_data['FAF'] <= 1.0:
            advice_list.append("- **운동량**이 부족합니다. 가벼운 산책부터 시작하세요.")
        if user_data['CAEC'] >= 2:
            advice_list.append("- **잦은 간식**이 비만의 원인일 수 있습니다. 간식을 줄여보세요.")
        if user_data['FAVC'] == 1:
            advice_list.append("- **고칼로리 음식(튀김, 인스턴트)** 섭취를 줄이시는 게 좋습니다.")
        if user_data['CALC'] >= 2:
            advice_list.append("- **잦은 음주**는 복부 비만의 원인입니다. 음주 횟수를 줄이세요.")

        if not advice_list:
            st.write("특별히 나쁜 습관이 보이지 않습니다. 훌륭한 자기관리 중이시네요! 👍")
        else:
            for advice in advice_list:
                st.write(advice)

        if st.button("🔄 다시 진단하기"):
            st.session_state['prediction_result'] = None
            st.rerun()