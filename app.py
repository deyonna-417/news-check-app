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

# --- ページ設定 ---
st.set_page_config(
    page_title="FactCheck Pro | Advanced Analytics",
    page_icon="🧪",
    layout="wide"
)

# --- カスタムCSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .report-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .metric-val { font-size: 24px; font-weight: bold; color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- ロジック ---
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return " ".join([text[i:i+2] for i in range(len(text)-1)])

# --- データ ---
CSV_DATA = """title,source
磐越道 高校生など21人死傷事故 バス運行会社を捜索,NHKニュース
イランと米 双方相手の攻撃主張 トランプ大統領「停戦有効」,NHKニュース
トランプ政権の10％関税「違法」と判断 米国際貿易裁判所,NHKニュース
関西電力 美浜原発3号機 蒸気漏れで運転停止 外部への影響なし,NHKニュース
株価 初の6万2000円台 イラン情勢の緊張緩和期待,NHKニュース
政府 独自の対北朝鮮制裁を2年延長 貨客船の入港禁止継続,読売新聞
リニア中央新幹線 静岡工区の着工 専門家会議で議論続く,産経新聞
都内の桜 満開を発表 平年より4日早く,共同通信
大阪万博 海外パビリオンの建設遅れ 対策を強化,日本経済新聞
AIによる画像生成 著作権保護の指針案を公表 文化庁,朝日新聞
"""
df = pd.read_csv(io.StringIO(CSV_DATA))

# --- メイン UI ---
st.title("🧪 FactCheck AI: 学術的解析モード")
st.markdown("演習・発表向け：Matplotlib/Seaborn を用いた詳細な統計分析表示")

with st.expander("📝 分析対象の入力", expanded=True):
    input_text = st.text_area("ニュース記事をペースト", height=150, placeholder="ここに文章を入力してください...")
    analyze = st.button("🔬 詳細統計分析を実行")

if analyze and input_text:
    with st.spinner('計算エンジン駆動中...'):
        # 計算
        titles = df["title"].tolist()
        words = [tokenize(t) for t in titles]
        input_words = tokenize(input_text)
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(words + [input_words])
        scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
        # 結果抽出
        df_res = df.copy()
        df_res["score"] = scores
        df_res = df_res.sort_values("score", ascending=False)
        top_score = df_res.iloc[0]["score"]
        
        # --- 表示セクション ---
        st.divider()
        
        col_main, col_stats = st.columns([1, 1])
        
        with col_main:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("🏁 総合判定レポート")
            if top_score > 0.15:
                st.success("### 【判定：信頼性高】")
                st.balloons()
            else:
                st.error("### 【判定：情報不足・未確認】")
            
            st.write(f"**最高一致度:** `{top_score:.4f}`")
            st.write(f"**参照元:** {df_res.iloc[0]['source']}")
            st.info(f"**最も近い報道:** {df_res.iloc[0]['title']}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Plotly メーター (動的要素)
            fig_m = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = top_score * 100,
                title = {'text': "一致率 (%)"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#58a6ff"}}
            ))
            fig_m.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_m, use_container_width=True)

        with col_stats:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("📊 統計的分布 (Seaborn)")
            
            # Seaborn を使ったヒストグラムと密度の可視化
            fig_sns, ax = plt.subplots(figsize=(5, 4))
            fig_sns.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            
            sns.histplot(scores, kde=True, color='#58a6ff', ax=ax)
            ax.set_title("Similarity Score Distribution", color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            
            st.pyplot(fig_sns)
            st.write("データベース内の全記事との類似度分布を表示。今回の入力がどの程度特異か、または普遍的かを示します。")
            st.markdown('</div>', unsafe_allow_html=True)

        # 下段：ヒートマップ (Seaborn)
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("🌡️ 類似度相関ヒートマップ (Top 5)")
        
        top_5 = df_res.head(5)
        # 簡易的なヒートマップ用行列
        hm_data = top_5[["score"]].T
        hm_data.columns = [t[:10]+"..." for t in top_5["title"]]
        
        fig_hm, ax_hm = plt.subplots(figsize=(10, 2))
        fig_hm.patch.set_facecolor('#161b22')
        sns.heatmap(hm_data, annot=True, cmap="YlGnBu", ax=ax_hm, cbar=False)
        ax_hm.set_title("Comparison Heatmap", color='white')
        ax_hm.tick_params(colors='white')
        st.pyplot(fig_hm)
        st.markdown('</div>', unsafe_allow_html=True)
