import streamlit as st
import pandas as pd
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob

# --- ページ設定 ---
st.set_page_config(page_title="FactChecker AI", page_icon="🛡️", layout="wide")

# --- 判定ロジック ---
def tokenize(text):
    try:
        tagger = MeCab.Tagger("-Owakati")
        result = tagger.parse(str(text))
        return result.strip() if result else str(text)
    except:
        return str(text)

def analyze_reliability(input_text, reference_df):
    input_words = tokenize(input_text)
    ref_words = reference_df['title'].astype(str).apply(tokenize).tolist()
    all_texts = ref_words + [input_words]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    max_score = similarities.max()
    best_match_index = similarities.argmax()
    
    return max_score, reference_df.iloc[best_match_index]['title']

# --- メイン処理 ---
st.title("🛡️ 誤情報拡散パターン分析ツール")

# 【重要】「nhkrss」という文字が含まれるCSVファイルを自動で探す
csv_files = glob.glob("*nhkrss*.csv")

df_nhk = pd.DataFrame()
if csv_files:
    # 一番新しいファイルを使う
    target_file = csv_files[0]
    try:
        # 文字化けに強い設定で読み込み
        df_nhk = pd.read_csv(target_file, encoding='utf-8-sig')
        st.sidebar.success(f"✅ 読み込み成功: {target_file}")
    except Exception as e:
        st.sidebar.error(f"❌ 読み込み失敗: {e}")
else:
    st.sidebar.warning("⚠️ フォルダ内に 'nhkrss' を含むCSVが見つかりません")

# 入力エリア
input_news = st.text_area("分析したいニュースを入力してください")

if st.button("信頼性を判定する"):
    if not df_nhk.empty and input_news:
        with st.spinner('分析中...'):
            score, best_match = analyze_reliability(input_news, df_nhk)
            
            if score > 0.3:
                st.success(f"判定ランク: S (スコア: {score:.2f})")
                st.write("信頼性が高い情報です。")
            elif score > 0.1:
                st.warning(f"判定ランク: A (スコア: {score:.2f})")
                st.write("話題は共通していますが、注意して確認してください。")
            else:
                st.error(f"判定ランク: B (スコア: {score:.2f})")
                st.write("注意が必要です。NHKのデータに一致する内容がありません。")
            
            with st.expander("最も近いNHKニュースを表示"):
                st.write(f"📌 {best_match}")