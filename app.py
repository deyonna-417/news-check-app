import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- ページ設定（実用的なビジネスツール風） ---
st.set_page_config(
    page_title="FactChecker Platform v1.0",
    page_icon="⚖️",
    layout="wide"
)

# --- カスタムCSS（清潔感のあるビジネスデザイン） ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; color: #1e1e1e; }
    .stButton>button { border-radius: 20px; }
    .action-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ロジック（N-gram解析） ---
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return " ".join([text[i:i+2] for i in range(len(text)-1)])

# --- データベース（実用性を意識した報道データ） ---
CSV_DATA = """title,source,url
【速報】都内で季節外れの夏日を記録 熱中症に警戒,NHKニュース,https://www3.nhk.or.jp/news/
新型iPhoneの発表は来月15日に決定か 米大手メディア報じる,IT Media,https://www.itmedia.co.jp/
マイナカードの健康保険証利用 2024年秋に一本化方針,朝日新聞,https://www.asahi.com/
生成AIの利用ガイドラインを文科省が公表 学校現場での活用,産経新聞,https://www.sankei.com/
プロ野球 日本シリーズが開幕 満員の中で熱戦,共同通信,https://www.kyodo.co.jp/
"""
df = pd.read_csv(io.StringIO(CSV_DATA))

# --- メイン UI ---
st.title("⚖️ FactChecker Platform v1.0")
st.write("報道機関のアーカイブとAI照合を行い、情報の真偽性をリアルタイムにスクリーニングします。")

# 1. ユーザー入力セクション
st.divider()
st.markdown("### 📥 1. 検証対象の入力")

# サンプルボタン（実用的なUX）
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    if st.button("📝 サンプル：本物のニュース"):
        st.session_state.text = "都内で夏日が記録され、熱中症への警戒が呼びかけられています。"
with col_s2:
    if st.button("❌ サンプル：未確認情報"):
        st.session_state.text = "都内の公園でライオンが逃げ出したという噂が広がっています。"

input_text = st.text_area("ニュース本文、またはSNSの投稿内容を入力してください", 
                          value=st.session_state.get('text', ''), 
                          height=150)

# 2. 分析実行
if st.button("🔬 信頼性スコアリングを開始"):
    if input_text:
        with st.spinner('検証データベースと照合中...'):
            # --- 解析処理 ---
            titles = df["title"].tolist()
            words = [tokenize(t) for t in titles]
            input_words = tokenize(input_text)
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(words + [input_words])
            scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
            
            best_idx = scores.argmax()
            top_score = scores[best_idx]
            match_data = df.iloc[best_idx]
            
            # --- 結果表示レイアウト ---
            st.divider()
            st.markdown("### 📊 2. 検証レポート")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown("#### 🏁 判定結果")
                if top_score > 0.15:
                    st.success(f"### 【判定：信頼性・高】\nこの情報は公的報道機関の内容と一致しています。")
                else:
                    st.error(f"### 【判定：信頼性・未確認】\n公的報道機関のデータベースに一致する情報が見当たりません。")
                
                # メーター表示
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = top_score * 100,
                    title = {'text': "報道一致率 (%)"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"}}
                ))
                fig_g.update_layout(height=250, margin=dict(t=30, b=0))
                st.plotly_chart(fig_g, use_container_width=True)

            with res_col2:
                st.markdown("#### 🔍 照合の詳細")
                st.write(f"**最も類似した記事:**\n{match_data['title']}")
                st.write(f"**情報源:** {match_data['source']}")
                st.markdown(f"[🔗 出典元リンクを開く]({match_data['url']})")
                
                # 統計グラフ (Seaborn)
                fig_sns, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=scores, y=[t[:10]+"..." for t in titles], ax=ax, palette="Blues_d")
                ax.set_title("Similarity comparison across database")
                st.pyplot(fig_sns)

            # 3. ユーザーへの行動指針（実用機能）
            st.divider()
            st.markdown("### 🛡️ 3. 推奨されるアクション")
            
            with st.container():
                st.markdown('<div class="action-box">', unsafe_allow_html=True)
                if top_score > 0.15:
                    st.write("✅ **このまま引用・拡散して問題ありません。**")
                    st.write("公的ソースが存在します。情報の詳細については、上記の出典元リンクを参照してください。")
                else:
                    st.write("🛑 **情報の拡散を一時停止してください。**")
                    st.write("SNS上でのみ拡散されている情報の可能性があります。信頼できる大手メディア（新聞、通信社など）から続報が出るまで待機してください。")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("検証する文章を入力してください。")
