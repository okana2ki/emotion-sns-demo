import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from datetime import datetime
import json
import os
import time

# ページ設定
st.set_page_config(page_title="感情分析SNS", page_icon="🎭", layout="wide")

# 共有データファイルのパス
DATA_FILE = "shared_posts.json"

def load_shared_posts():
    """共有投稿データを読み込み"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # datetime文字列をdatetimeオブジェクトに変換
                for post in data:
                    post['time'] = datetime.fromisoformat(post['time'])
                return data
        else:
            return []
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return []

def save_shared_posts(posts):
    """共有投稿データを保存"""
    try:
        # datetimeオブジェクトを文字列に変換
        data_to_save = []
        for post in posts:
            post_copy = post.copy()
            post_copy['time'] = post['time'].isoformat()
            data_to_save.append(post_copy)
        
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"データ保存エラー: {e}")
        return False

def add_post(user_name, text, sentiment_score, emotion, color):
    """新しい投稿を追加"""
    posts = load_shared_posts()
    new_post = {
        'user': user_name,
        'text': text,
        'sentiment': sentiment_score,
        'emotion': emotion,
        'time': datetime.now(),
        'color': color,
        'id': len(posts) + 1
    }
    posts.append(new_post)
    return save_shared_posts(posts)

# メインタイトル
st.title("🎭 リアルタイム感情分析SNS")
st.markdown("**AIが投稿の感情を瞬時に分析 - 全員の投稿をリアルタイム共有！**")

# 自動更新設定
auto_refresh = st.checkbox("🔄 自動更新（5秒間隔）", value=True)

# レイアウト: 2列構成
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 投稿エリア")
    
    # ユーザー名入力
    user_name = st.text_input("ニックネーム", placeholder="未来の○○大生", key="username")
    
    # 投稿内容入力
    user_input = st.text_area(
        "今の気持ちを投稿してください",
        placeholder="例: 今日のオープンキャンパス楽しい！",
        height=100,
        key="post_input"
    )
    
    if user_input and user_name:
        # 感情分析（日本語・英語両対応）
        blob = TextBlob(user_input)
        sentiment_polarity = blob.sentiment.polarity
        
        # 日本語の場合の簡易感情分析補正
        positive_words = ['楽しい', '嬉しい', '最高', '良い', 'すごい', 'がんばる', '頑張る', '感動', '素晴らしい', 'ありがとう', '大好き', '幸せ']
        negative_words = ['悲しい', '辛い', '大変', '不安', '心配', '疲れた', 'つまらない', '嫌', '困った', 'ダメ']
        
        # 日本語キーワードベースの補正
        positive_count = sum(1 for word in positive_words if word in user_input)
        negative_count = sum(1 for word in negative_words if word in user_input)
        
        # 補正値計算
        keyword_adjustment = (positive_count - negative_count) * 0.3
        sentiment_polarity = max(-1, min(1, sentiment_polarity + keyword_adjustment))
        
        sentiment_score = (sentiment_polarity + 1) * 50
        
        # 感情判定
        if sentiment_score > 65:
            emotion = "😊 とてもポジティブ"
            color = "#28a745"
        elif sentiment_score > 55:
            emotion = "🙂 ポジティブ"
            color = "#17a2b8"
        elif sentiment_score > 45:
            emotion = "😐 ニュートラル"
            color = "#6c757d"
        elif sentiment_score > 35:
            emotion = "😞 ネガティブ"
            color = "#fd7e14"
        else:
            emotion = "😢 とてもネガティブ"
            color = "#dc3545"
        
        # リアルタイム感情分析結果表示
        st.markdown("### 🤖 AI分析結果")
        
        # メトリック表示
        st.metric(
            label="感情スコア",
            value=f"{sentiment_score:.1f}点",
            delta=f"{emotion}",
        )
        
        # プログレスバー
        st.markdown(f"""
        <div style="background-color: {color}; height: 20px; border-radius: 10px; width: {sentiment_score}%; margin: 10px 0;">
            <div style="color: white; text-align: center; line-height: 20px; font-weight: bold;">
                {sentiment_score:.1f}点
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 投稿ボタン
        if st.button("🚀 投稿する", type="primary"):
            if add_post(user_name, user_input, sentiment_score, emotion, color):
                st.success("投稿完了！全員が見ることができます 🎉")
                time.sleep(1)
                st.rerun()
            else:
                st.error("投稿に失敗しました。もう一度お試しください。")

with col2:
    st.subheader("🌟 みんなの投稿タイムライン")
    
    # 手動更新ボタン
    if st.button("🔄 最新を取得"):
        st.rerun()
    
    # 共有投稿を読み込み
    all_posts = load_shared_posts()
    
    # 投稿がある場合のみ表示
    if all_posts:
        # 統計情報
        total_posts = len(all_posts)
        avg_sentiment = sum(post['sentiment'] for post in all_posts) / total_posts
        
        st.markdown("### 📊 リアルタイム統計")
        
        # 3つのメトリック
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("総投稿数", f"{total_posts}件")
        with metric_col2:
            st.metric("平均感情", f"{avg_sentiment:.1f}点")
        with metric_col3:
            positive_ratio = len([p for p in all_posts if p['sentiment'] > 55]) / total_posts * 100
            st.metric("ポジティブ率", f"{positive_ratio:.0f}%")
        
        # 最新投稿の通知
        if total_posts > 0:
            latest_post = all_posts[-1]
            time_diff = datetime.now() - latest_post['time']
            if time_diff.total_seconds() < 30:  # 30秒以内の投稿
                st.success(f"🔥 新着: {latest_post['user']}さんが投稿しました！")
        
        # タイムライン表示（最新から8件）
        st.markdown("### 💬 最新の投稿")
        display_posts = sorted(all_posts, key=lambda x: x['time'], reverse=True)[:8]
        
        for post in display_posts:
            time_ago = datetime.now() - post['time']
            if time_ago.total_seconds() < 60:
                time_str = f"{int(time_ago.total_seconds())}秒前"
            elif time_ago.total_seconds() < 3600:
                time_str = f"{int(time_ago.total_seconds() / 60)}分前"
            else:
                time_str = post['time'].strftime('%H:%M')
            
            with st.container():
                st.markdown(f"""
                <div style="
                    border-left: 4px solid {post['color']}; 
                    padding: 12px; 
                    margin: 8px 0; 
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>👤 {post['user']}</strong> 
                        <small style="color: #666;">⏰ {time_str}</small>
                    </div>
                    <div style="color: {post['color']}; font-weight: bold; margin: 5px 0;">
                        {post['emotion']} ({post['sentiment']:.1f}点)
                    </div>
                    <div style="margin-top: 8px; font-size: 16px;">
                        📝 {post['text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 感情推移グラフ
        if len(all_posts) > 1:
            st.markdown("### 📈 感情の推移（全体）")
            df = pd.DataFrame(all_posts)
            df['投稿順序'] = range(1, len(df) + 1)
            
            fig = px.line(
                df, 
                x='投稿順序', 
                y='sentiment',
                title='みんなの感情スコア変化',
                markers=True,
                hover_data=['user', 'text']
            )
            fig.update_layout(
                yaxis_title="感情スコア",
                xaxis_title="投稿順序",
                height=300
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                         annotation_text="ニュートラル(50点)")
            st.plotly_chart(fig, use_container_width=True)
            
            # 感情分布
            st.markdown("### 🎭 感情分布")
            emotion_counts = pd.DataFrame(all_posts)['emotion'].value_counts()
            fig2 = px.pie(values=emotion_counts.values, names=emotion_counts.index, 
                         title="投稿の感情分布")
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("まだ投稿がありません。左側から投稿してみてください！")

# 自動更新機能
if auto_refresh:
    time.sleep(5)
    st.rerun()

# サイドバーに情報
with st.sidebar:
    st.markdown("## 🎯 使い方")
    st.markdown("""
    1. **ニックネーム**を入力
    2. **気持ち**を文章で投稿
    3. **AI**が感情を0-100点で分析
    4. **全員**がリアルタイムで共有
    """)
    
    st.markdown("## 🌐 リアルタイム共有")
    st.markdown("""
    このアプリは**ファイルベース**で
    全ユーザーの投稿を共有しています。
    
    他の人の投稿もリアルタイムで
    表示されます！
    """)
    
    st.markdown("## 🤖 AI技術")
    st.markdown("""
    - **TextBlob**: 英語感情分析
    - **キーワード分析**: 日本語対応
    - **リアルタイム処理**: 瞬時に結果表示
    """)
    
    # 接続情報表示
    st.markdown("## 📱 アクセス情報")
    st.markdown(f"""
    **ローカル**: http://localhost:8501
    
    **ネットワーク**: http://192.168.2.102:8501
    
    同じWiFiに接続している人は
    ネットワークURLでアクセス可能！
    """)
    
    # 現在のデータ状況
    current_posts = load_shared_posts()
    st.markdown(f"### 📊 現在の状況")
    st.markdown(f"**総投稿数**: {len(current_posts)}件")
    if current_posts:
        latest = current_posts[-1]
        st.markdown(f"**最新投稿**: {latest['user']}さん")
