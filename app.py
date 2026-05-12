import streamlit as st
import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- ページ設定 ---
st.set_page_config(page_title="FactCheck AI", page_icon="🛡️", layout="wide")

# --- 判定ロジック (MeCabを使わない安全設計) ---
def safe_tokenize(text):
    text = str(text)
    # 2文字ずつに区切る「N-gram」という手法（これならエラーが出ません）
    return " ".join([text[i:i+2] for i in range(len(text)-1)])

# --- データの準備 ---
CSV_DATA = """title,source
磐越道 高校生など21人死傷事故 バス運行会社を捜索,NHKニュース
イランと米 双方相手の攻撃主張 トランプ大統領「停戦有効」,NHKニュース
トランプ政権の10％関税「違法」と判断 米国際貿易裁判所,NHKニュース
関西電力 美浜原発3号機 蒸気漏れで運転停止 外部への影響なし,NHKニュース
株価 初の6万2000円台 イラン情勢の緊張緩和期待,NHKニュース
"""
df_nhk = pd.read_csv(io.StringIO(CSV_DATA))

# --- メイン画面 ---
st.title("🛡️ FactCheck AI 分析システム")
st.write("公的な報道アーカイブと照合し、ニュースの信頼性を判定します。")

input_news = st.text_area("分析したいニュースを貼り付けてください", height=150)

if st.button("信頼性を判定する"):
    if input_news:
        with st.spinner('解析中...'):
            ref_titles = df_nhk["title"].tolist()
            ref_words = [safe_tokenize(t) for t in ref_titles]
            input_words = safe_tokenize(input_news)
            
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(ref_words + [input_words])
            sim = cosine_similarity(tfidf[-1], tfidf[:-1])
            
            score = sim.max()
            best_match = ref_titles[sim.argmax()]
            
            st.divider()
            if score > 0.2:
                st.success(f"### 【判定：S】 信頼性が高い情報です")
            else:
                st.error(f"### 【判定：B】 未確認の情報です")
            
            st.write(f"**AI一致スコア:** {score:.2f}")
            st.info(f"**最も近い公的報道:** {best_match}")
    else:
        st.warning("ニュースを入力してください。")
