import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- ページ設定 ---
st.set_page_config(page_title="News Verifier Pro", page_icon="🕵️", layout="wide")

# --- カスタムCSS（少し見た目を整える） ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 判定ロジック ---
def safe_tokenize(text):
    text = str(text)
    return " ".join([text[i:i+2] for i in range(len(text)-1)])

# --- データの準備（ここを増やすとさらに本格的に！） ---
CSV_DATA = """title,source
磐越道 高校生など21人死傷事故 バス運行会社を捜索,NHKニュース
イランと米 双方相手の攻撃主張 トランプ大統領「停戦有効」,NHKニュース
トランプ政権の10％関税「違法」と判断 米国際貿易裁判所,NHKニュース
関西電力 美浜原発3号機 蒸気漏れで運転停止 外部への影響なし,NHKニュース
株価 初の6万2000円台 イラン情勢の緊張緩和期待,NHKニュース
政府 独自の対北朝鮮制裁を2年延長 貨客船の入港禁止継続,読売新聞
リニア中央新幹線 静岡工区の着工 専門家会議で議論続く,産経新聞
"""
df_nhk = pd.read_csv(io.StringIO(CSV_DATA))

# --- サイドバー ---
st.sidebar.header("📊 システム設定")
threshold = st.sidebar.slider("判定しきい値", 0.1, 0.5, 0.2)
st.sidebar.write("この値を超えると『信頼性が高い』と判定されます。")

# --- メインコンテンツ ---
st.title("🕵️ News Verifier Pro")
st.subheader("AIによる公的報道照合・信頼性分析システム")

input_news = st.text_area("分析するニュース記事を入力してください", placeholder="ここに文章をペースト...", height=150)

if st.button("🔍 分析実行"):
    if input_news:
        with st.spinner('AIがデータベースを照合中...'):
            # 計算処理
            ref_titles = df_nhk["title"].tolist()
            ref_words = [safe_tokenize(t) for t in ref_titles]
            input_words = safe_tokenize(input_news)
            
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(ref_words + [input_words])
            sim_scores = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
            
            # 上位の結果を取得
            best_idx = sim_scores.argmax()
            max_score = sim_scores[best_idx]
            
            # 信頼度を100%換算に調整（見栄えのため）
            confidence_score = min(max_score * 2.5, 1.0) * 100 

            # --- 結果表示レイアウト ---
            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("### 🛡️ 信頼度メーター")
                # メーターグラフ（ゲージ）
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Reliability Score (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ffcccc"},
                            {'range': [40, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#d4edda"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 250
                        }
                    }
                ))
                fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.write("### 📈 類似ニュース比較 (Top 3)")
                # 類似度TOP3の棒グラフ
                top_indices = sim_scores.argsort()[-3:][::-1]
                top_data = pd.DataFrame({
                    "News Title": [ref_titles[i][:20] + "..." for i in top_indices],
                    "Score": [sim_scores[i] for i in top_indices]
                })
                fig_bar = px.bar(top_data, x='Score', y='News Title', orientation='h',
                                 color='Score', color_continuous_scale='Blues')
                fig_bar.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            # --- 最終判定 ---
            st.divider()
            if max_score > threshold:
                st.success(f"✅ **判定結果：【信頼性・高】**")
                st.balloons()
            else:
                st.error(f"⚠️ **判定結果：【信頼性・未確認】**")

            st.info(f"**最も類似した公的ソース:** {ref_titles[best_idx]}")
            
    else:
        st.warning("ニュース文章を入力してください。")
