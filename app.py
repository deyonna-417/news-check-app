import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- ページ設定 ---
st.set_page_config(page_title="FactCheck AI Pro", page_icon="⚖️", layout="wide")

# --- データの読み込み (GitHubのCSVを読み込む) ---
@st.cache_data
def load_data():
    # NHKの正解データ（手動定義）
    nhk_data = """title,source,url
磐越道 高校生など21人死傷事故 バス運行会社を捜索,NHKニュース,https://www3.nhk.or.jp/news/
イランと米 双方相手の攻撃主張 トランプ大統領「停戦有効」,NHKニュース,https://www3.nhk.or.jp/news/
トランプ政権の10％関税「違法」と判断 米国際貿易裁判所,NHKニュース,https://www3.nhk.or.jp/news/
関西電力 美浜原発3号機 蒸気漏れで運転停止 外部への影響なし,NHKニュース,https://www3.nhk.or.jp/news/
    """
    df_nhk = pd.read_csv(io.StringIO(nhk_data))
    df_nhk['label'] = '正解 (NHK)'

    try:
        # GitHubにアップしたCSVを読み込む
        df_gnews = pd.read_csv("news_health.csv")
        # GNews特有のソース名抽出処理
        if 'source' in df_gnews.columns:
            df_gnews['source'] = df_gnews['source'].apply(lambda x: eval(x)['name'] if (isinstance(x, str) and '{' in x) else x)
        df_gnews['label'] = '検証対象 (GNews)'
        # 合体させる
        return pd.concat([df_nhk, df_gnews], ignore_index=True)
    except:
        return df_nhk

df = load_data()

# --- ロジック ---
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return " ".join([text[i:i+2] for i in range(len(text)-1)])

# --- UI ---
st.title("⚖️ FactCheck AI: 実証実験プラットフォーム")
st.markdown("### 目的：公共放送(NHK)を正解とし、ネットニュース(GNews)の信頼性を検証する")

# 入力
input_text = st.text_area("検証したいニュース記事を入力してください", height=150)

if st.button("🔬 信頼性分析を実行") and input_text:
    # 計算
    titles = df["title"].tolist()
    words = [tokenize(t) for t in titles]
    input_words = tokenize(input_text)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(words + [input_words])
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # スコア付与
    df_res = df.copy()
    df_res['score'] = scores
    df_res = df_res.sort_values('score', ascending=False)
    
    # 最上位の結果
    best = df_res.iloc[0]
    
    # レイアウト表示
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏁 判定レポート")
        # NHKのデータと一致するかで判定
        is_nhk_match = (best['label'] == '正解 (NHK)' and best['score'] > 0.15)
        
        if is_nhk_match:
            st.success("### ✅ 【判定：信頼性・高】\nNHKの報道内容と高い一致を確認しました。")
            st.balloons()
        else:
            st.error("### ⚠️ 【判定：信頼性・未確認】\nNHKの報道ベースでは確認できない内容です。独自情報の可能性があります。")
        
        st.metric("最高一致スコア", f"{best['score']:.3f}")
        st.write(f"**照合先:** {best['title']}")
        st.write(f"**ソース:** {best['source']} ({best['label']})")

    with col2:
        st.subheader("📊 統計分析（重複・類似分布）")
        # Seabornで全データとの類似度分布を表示
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(scores, kde=True, color="skyblue", ax=ax)
        ax.set_title("Similarity Distribution across Database")
        st.pyplot(fig)
        st.caption("データベース内の全記事（重複含む）との一致度分布。山が高いほど似た記事が多いことを示します。")

    # 3. アクション
    st.info(f"👉 **推奨アクション:** {'このまま引用可能です。' if is_nhk_match else '複数の大手メディアで続報を確認するまで拡散を控えてください。'}")
