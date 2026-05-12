import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- ページ設定（プロ仕様のダークテーマ） ---
st.set_page_config(
    page_title="News Verifier Pro | Dark Edition",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- カスタムCSS（ダークモードとカードデザイン） ---
st.markdown("""
    <style>
    /* 全体の背景色 */
    .stApp { background-color: #1a1d21; color: #e1e4e8; }
    
    /* サイドバー */
    .css-1d391kg { background-color: #111316; }
    
    /* カードデザイン（判定結果、メーター、グラフ） */
    .analysis-card {
        background-color: #23272e;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* タイトル周りの装飾 */
    h1, h2, h3 { color: #ffffff; font-weight: 700; }
    .main-title {
        background: linear-gradient(90deg, #1f77b4 0%, #61dafb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
    }
    
    /* ボタンのカスタマイズ */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4 0%, #4a90e2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #4a90e2 0%, #1f77b4 100%); box-shadow: 0 0 15px rgba(74, 144, 226, 0.5); }
    </style>
    """, unsafe_allow_html=True)

# --- 判定ロジック ---
def safe_tokenize(text):
    text = str(text)
    # 記号を除去して2文字ずつに区切る
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join([text[i:i+2] for i in range(len(text)-1)])

# --- データの準備 ---
CSV_DATA = """title,source
磐越道 高校生など21人死傷事故 バス運行会社を捜索,NHKニュース
イランと米 双方相手の攻撃主張 トランプ大統領「停戦有効」,NHKニュース
トランプ政権の10％関税「違法」と判断 米国際貿易裁判所,NHKニュース
関西電力 美浜原発3号機 蒸気漏れで運転停止 外部への影響なし,NHKニュース
株価 初の6万2000円台 イラン情勢の緊張緩和期待,NHKニュース
政府 独自の対北朝鮮制裁を2年延長 貨客船の入港禁止継続,読売新聞
リニア中央新幹線 静岡工区の着工 専門家会議で議論続く,産経新聞
都内の桜 満開を発表 平年より4日早く,共同通信
"""
df_nhk = pd.read_csv(io.StringIO(CSV_DATA))

# --- サイドバー ---
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/shield.png", width=50)
    st.markdown("### プロフェッショナル・ニュース分析官")
    st.divider()
    st.header("📊 システム設定")
    threshold = st.slider("判定しきい値", 0.05, 0.40, 0.15, 0.01)
    st.info(f"現在のしきい値: {threshold:.2f}")
    st.markdown("""
        **【しきい値の目安】**
        * **0.10**: かなり緩い判定
        * **0.15**: 標準的な判定（おすすめ）
        * **0.20以上**: 非常に厳格な判定
    """)

# --- メインコンテンツ ---
st.markdown('<h1 class="main-title">News Verifier Pro</h1>', unsafe_allow_html=True)
st.subheader("🕵️ 次世代AIによる公的報道照合・信頼性分析プラットフォーム")
st.markdown("---")

# 入力セクション
st.markdown("### 1. ニュース記事を入力")
input_news = st.text_area("", placeholder="分析したいニュース文章、SNSの投稿などを貼り付けてください...", height=200)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    analyze_btn = st.button("🚨 リアルタイム分析を実行")

# --- 分析結果セクション（ボタンが押されたら表示） ---
if analyze_btn:
    if input_news:
        with st.spinner('🚀 AIがデータベースと照合中... (N-gram TF-IDF計算開始)'):
            # 計算処理
            ref_titles = df_nhk["title"].tolist()
            ref_words = [safe_tokenize(t) for t in ref_titles]
            input_words = safe_tokenize(input_news)
            
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(ref_words + [input_words])
            sim_scores = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
            
            best_idx = sim_scores.argmax()
            max_score = sim_scores[best_idx]
            best_title = ref_titles[best_idx]
            
            # --- 根拠の可視化 (単語の一致チェック) ---
            # 入力文を2文字ずつに分解し、上位ニュースに含まれるかを判定
            tokens = input_words.split()
            matched_tokens = []
            for token in tokens:
                if token in ref_words[best_idx]:
                    matched_tokens.append((token, 1)) # 一致した
                else:
                    matched_tokens.append((token, 0)) # 一致してない
            
            # --- 結果表示レイアウト ---
            st.divider()
            st.markdown("### 2. 分析結果レポート")

            # 判定結果カード
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            if max_score > threshold:
                st.success(f"## ✅ **判定：【信頼性・高】**")
                st.balloons()
            else:
                st.error(f"## ⚠️ **判定：【信頼性・未確認】**")
            st.info(f"**最も類似した公的ソース:** {best_title}")
            st.markdown(f"**AI一致スコア:** `{max_score:.3f}` (しきい値: `{threshold:.2f}`)")
            st.markdown('</div>', unsafe_allow_html=True)

            # メーターとグラフのカード
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.write("#### 🛡️ 信頼度メーター")
                # ダークモード用メーターグラフ
                confidence_score = min(max_score * 3.0, 1.0) * 100 # 見栄えのための調整
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#fff"},
                        'bar': {'color': "#61dafb"}, # 青白い光のような色
                        'bgcolor': "#1a1d21",
                        'borderwidth': 2,
                        'bordercolor': "#30363d",
                        'steps': [
                            {'range': [0, threshold*300], 'color': "#30363d"}, # しきい値未満はダークグレー
                            {'range': [threshold*300, 100], 'color': "#1a5a8a"}  # しきい値以上は青
                        ],
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.write("#### 📈 類似ニュース比較 (Top 3)")
                # ダークモード用棒グラフ
                top_indices = sim_scores.argsort()[-3:][::-1]
                top_data = pd.DataFrame({
                    "News Title": [ref_titles[i][:15] + "..." for i in top_indices],
                    "Score": [sim_scores[i] for i in top_indices]
                })
                # scoreに基づいて色を決定
                color_map = [ "#61dafb" if s > threshold else "#4a4a4a" for s in top_data["Score"]]
                fig_bar = go.Figure(go.Bar(
                    x=top_data["Score"],
                    y=top_data["News Title"],
                    orientation='h',
                    marker_color=color_map,
                    text=[f"{s:.3f}" for s in top_data["Score"]],
                    textposition='inside',
                    textfont={'color': 'black'}
                ))
                fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, xaxis={'gridcolor': "#30363d"})
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ヒートマップ（根拠の可視化）カード
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.write("#### 🔍 判定の根拠（入力文と上位ニュースの単語一致）")
            # 一致した単語を色付きで表示（ヒートマップの簡易版）
            st.write("公的報道と**一致した単語（2文字単位）**を色分けして表示しています：")
            
            html_text = '<div style="background-color: #1a1d21; padding: 15px; border-radius: 8px; border: 1px solid #30363d;">'
            for token, is_matched in matched_tokens:
                if is_matched:
                    # 一致した単語は青背景に白文字
                    html_text += f'<span style="background-color: #1f77b4; color: white; padding: 2px 4px; border-radius: 4px; margin: 2px; display: inline-block;">{token}</span>'
                else:
                    # 一致してない単語は薄いグレー
                    html_text += f'<span style="color: #8b949e; margin: 2px; display: inline-block;">{token}</span>'
            html_text += '</div>'
            st.markdown(html_text, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.warning("ニュース文章を入力してください。")
