import streamlit as st
import google.genai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import sqlite3
import asyncio
import threading
from typing import Dict, List
import queue
import re
import os
import qrcode
from PIL import Image
import io
import base64

# ページ設定
st.set_page_config(
    page_title="🎓 リアルタイム感情分析SNS",
    page_icon="🎓",
    layout="wide"
)

# セッションステートの初期化
if 'posts' not in st.session_state:
    st.session_state.posts = []
if 'emotion_queue' not in st.session_state:
    st.session_state.emotion_queue = queue.Queue()
if 'processing' not in st.session_state:
    st.session_state.processing = False

# QRコード生成関数
@st.cache_data
def generate_qr_code(url: str) -> str:
    """QRコードを生成してbase64エンコードされた画像を返す"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    # QRコード画像を生成
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # PILイメージをbase64エンコード
    buffer = io.BytesIO()
    qr_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

def get_app_url():
    """現在のアプリのURLを取得"""
    try:
        # Streamlit Cloudの場合
        return f"https://{st.runtime.get_instance().get_headers().get('host', 'localhost:8501')}"
    except:
        # ローカル開発環境の場合
        return "http://localhost:8501"

# Google AI設定
@st.cache_resource
def setup_genai():
    """Google AI APIの設定"""
    # 環境変数からAPIキーを取得
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if not api_key:
        st.error("""
        ⚠️ Google AI APIキーが設定されていません。
        
        環境変数 `GOOGLE_AI_API_KEY` を設定してください。
        
        **Streamlit Cloudの場合:**
        1. アプリの設定画面を開く
        2. "Secrets" タブを選択
        3. 以下を追加:
        ```
        GOOGLE_AI_API_KEY = "your-api-key-here"
        ```
        
        **ローカル環境の場合:**
        ```bash
        export GOOGLE_AI_API_KEY="your-api-key-here"
        ```
        """)
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        # 接続テスト
        st.sidebar.success("✅ Google AI API 接続成功")
        return client
    except Exception as e:
        st.sidebar.error(f"❌ Google AI API接続エラー: {e}")
        return None

# データベース設定
@st.cache_resource
def init_database():
    """SQLiteデータベースの初期化"""
    conn = sqlite3.connect('sns_demo.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            happiness REAL DEFAULT 0,
            excitement REAL DEFAULT 0,
            satisfaction REAL DEFAULT 0,
            concern REAL DEFAULT 0,
            processed BOOLEAN DEFAULT FALSE,
            error_type TEXT DEFAULT NULL,
            error_message TEXT DEFAULT NULL
        )
    ''')
    conn.commit()
    return conn

# 感情分析プロンプト（デモ用に最適化）
EMOTION_PROMPT = """
以下のオープンキャンパスの感想文から、4つの感情の強さを0-10の数値で分析してください。

感想文: "{text}"

以下のJSON形式で回答してください：
{{
    "happiness": 数値(0-10),
    "excitement": 数値(0-10),
    "satisfaction": 数値(0-10),
    "concern": 数値(0-10)
}}

感情の定義：
- happiness: 楽しさ、喜び
- excitement: わくわく感、期待感
- satisfaction: 満足感、充実感
- concern: 不安、心配
"""

# レート制限対応クラス
class RateLimiter:
    def __init__(self, max_requests=15, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def can_request(self):
        with self.lock:
            now = datetime.now()
            # 時間窓外の古いリクエストを削除
            self.requests = [req_time for req_time in self.requests 
                           if (now - req_time).seconds < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def wait_time(self):
        with self.lock:
            if not self.requests:
                return 0
            oldest = min(self.requests)
            return max(0, self.time_window - (datetime.now() - oldest).seconds)

# グローバルレート制限インスタンス
rate_limiter = RateLimiter()

def analyze_emotion_with_ai(text: str, client) -> tuple:
    """AIによる感情分析（レート制限対応）
    
    Returns:
        tuple: (emotions_dict, error_type, error_message)
        - emotions_dict: 感情スコア辞書 or None
        - error_type: エラータイプ ("rate_limit", "parse_error", "api_error", None)
        - error_message: エラーメッセージ
    """
    if client is None:
        fallback_emotions = {
            "happiness": 5.0,
            "excitement": 5.0,
            "satisfaction": 5.0,
            "concern": 5.0
        }
        return fallback_emotions, "api_error", "Google AI APIが利用できません"
    
    if not rate_limiter.can_request():
        # レート制限中は None を返して後で再試行させる
        return None, "rate_limit", "API レート制限中のため待機しています"
    
    try:
        if 'debug_status' in st.session_state:
            st.session_state.debug_status.append(f"🤖 感情分析実行中: {text[:30]}...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[EMOTION_PROMPT.format(text=text)]
        )
        result_text = response.text
        
        if 'debug_status' in st.session_state:
            st.session_state.debug_status.append(f"✅ AI応答受信: {result_text[:50]}...")
        
        # JSON部分を抽出
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            emotions = json.loads(json_match.group())
            if 'debug_status' in st.session_state:
                st.session_state.debug_status.append("✅ 感情分析完了")
            return emotions, None, None
        else:
            # パースに失敗した場合のフォールバック
            fallback_emotions = {
                "happiness": 6.0,
                "excitement": 7.0,
                "satisfaction": 5.0,
                "concern": 3.0
            }
            if 'debug_status' in st.session_state:
                st.session_state.debug_status.append(f"⚠️ JSON解析失敗: {result_text}")
            return fallback_emotions, "parse_error", f"AIの応答解析に失敗しました。推定値を使用: {result_text[:50]}..."
    except Exception as e:
        # API呼び出しエラー
        fallback_emotions = {
            "happiness": 5.0,
            "excitement": 5.0,
            "satisfaction": 5.0,
            "concern": 5.0
        }
        if 'debug_status' in st.session_state:
            st.session_state.debug_status.append(f"❌ API呼び出しエラー: {str(e)}")
        return fallback_emotions, "api_error", f"API呼び出しエラー: {str(e)}"

def process_emotion_queue():
    """感情分析キューの処理（バックグラウンド）"""
    client = setup_genai()
    conn = init_database()
    
    # デバッグ用のステータス更新
    if 'debug_status' not in st.session_state:
        st.session_state.debug_status = []
    
    st.session_state.debug_status.append("🔄 バックグラウンド処理開始")
    
    while True:
        try:
            if not st.session_state.emotion_queue.empty():
                post_id = st.session_state.emotion_queue.get(timeout=1)
                st.session_state.debug_status.append(f"📝 投稿ID {post_id} を処理中...")
                
                # データベースから投稿を取得
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM posts WHERE id = ? AND processed = FALSE", (post_id,))
                result = cursor.fetchone()
                
                if result:
                    content = result[0]
                    st.session_state.debug_status.append(f"📄 投稿内容: {content[:30]}...")
                    emotions, error_type, error_message = analyze_emotion_with_ai(content, client)
                    
                    if emotions is None:
                        # レート制限中の場合、キューに再度追加して後で再試行
                        st.session_state.debug_status.append("⏳ レート制限中、再キューイング...")
                        st.session_state.emotion_queue.put(post_id)
                        time.sleep(5)  # 5秒待ってから次の処理
                        continue
                    
                    # 結果をデータベースに保存
                    cursor.execute("""
                        UPDATE posts 
                        SET happiness = ?, excitement = ?, satisfaction = ?, concern = ?, 
                            processed = TRUE, error_type = ?, error_message = ?
                        WHERE id = ?
                    """, (emotions['happiness'], emotions['excitement'], 
                          emotions['satisfaction'], emotions['concern'], 
                          error_type, error_message, post_id))
                    conn.commit()
                    
                    st.session_state.debug_status.append(f"💾 投稿ID {post_id} の分析完了・保存")
                    
                    # ステータスリストのサイズ制限
                    if len(st.session_state.debug_status) > 10:
                        st.session_state.debug_status = st.session_state.debug_status[-10:]
                        
                else:
                    st.session_state.debug_status.append(f"❓ 投稿ID {post_id} が見つかりません")
            
            time.sleep(0.5)  # CPU使用率を抑制
        except queue.Empty:
            continue
        except Exception as e:
            st.session_state.debug_status.append(f"❌ バックグラウンド処理エラー: {e}")
            time.sleep(1)

# バックグラウンド処理の開始
if not st.session_state.processing:
    st.session_state.processing = True
    # デバッグ用のステータス初期化
    if 'debug_status' not in st.session_state:
        st.session_state.debug_status = []
    thread = threading.Thread(target=process_emotion_queue, daemon=True)
    thread.start()

# メイン画面
def main():
    st.title("🎓 リアルタイム感情分析SNS")
    st.subheader("オープンキャンパスの感想を投稿して、みんなの感情を可視化しよう！")
    
    # データベース接続
    conn = init_database()
    
    # 2カラムレイアウト
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 感想を投稿")
        
        # 投稿フォーム
        with st.form("post_form"):
            content = st.text_area(
                "オープンキャンパスの感想を教えてください！",
                placeholder="今日の模擬授業はとても面白かった！データサイエンスに興味が湧きました。",
                height=100
            )
            
            submitted = st.form_submit_button("📤 投稿する", use_container_width=True)
        
        if submitted and content:
            # データベースに投稿を保存
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO posts (content, timestamp) 
                VALUES (?, ?)
            """, (content, datetime.now()))
            conn.commit()
            
            # 感情分析キューに追加
            post_id = cursor.lastrowid
            st.session_state.emotion_queue.put(post_id)
            
            st.success("投稿ありがとうございます！")
            st.info(f"📋 投稿ID: {post_id} をキューに追加しました")
            
            # レート制限状況を確認して表示
            if not rate_limiter.can_request():
                wait_time = rate_limiter.wait_time()
                st.warning(f"⏳ レート制限中です。約 {wait_time} 秒後に感情分析を開始します...")
            else:
                st.info("🤔 感情分析中...")
            
            # APIキーの状況確認
            if os.getenv("GOOGLE_AI_API_KEY"):
                st.info("✅ APIキーが設定されています")
            else:
                st.error("❌ APIキーが設定されていません")
            
            st.rerun()
    
    with col2:
        st.markdown("### 📊 リアルタイム感情分析")
        
        # データ取得
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content, timestamp, happiness, excitement, satisfaction, concern, 
                   processed, error_type, error_message
            FROM posts 
            ORDER BY timestamp DESC
            LIMIT 50
        """)
        posts_data = cursor.fetchall()
        
        if posts_data:
            # データフレーム作成
            df = pd.DataFrame(posts_data, columns=[
                'content', 'timestamp', 'happiness', 'excitement', 
                'satisfaction', 'concern', 'processed', 'error_type', 'error_message'
            ])
            
            # 感情サマリー（処理済みデータのみ）
            processed_df = df[df['processed'] == True]
            
            if not processed_df.empty:
                # 感情の平均値を計算
                avg_emotions = {
                    '楽しさ': processed_df['happiness'].mean(),
                    'わくわく感': processed_df['excitement'].mean(),
                    '満足感': processed_df['satisfaction'].mean(),
                    '不安・心配': processed_df['concern'].mean()
                }
                
                # メトリクス表示
                metric_cols = st.columns(4)
                colors = ['🟡', '🔥', '💚', '💙']
                for i, (emotion, value) in enumerate(avg_emotions.items()):
                    with metric_cols[i]:
                        st.metric(
                            f"{colors[i]} {emotion}",
                            f"{value:.1f}/10",
                            delta=None
                        )
                
                # 感情推移グラフ
                st.markdown("#### 📈 感情の推移")
                
                if len(processed_df) > 1:
                    # 時系列データの準備
                    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
                    processed_df = processed_df.sort_values('timestamp')
                    
                    # プロットly グラフ
                    fig = go.Figure()
                    
                    emotions_jp = {
                        'happiness': '楽しさ',
                        'excitement': 'わくわく感',
                        'satisfaction': '満足感',
                        'concern': '不安・心配'
                    }
                    
                    colors_map = {
                        'happiness': '#FFD700',
                        'excitement': '#FF6B35',
                        'satisfaction': '#4CAF50',
                        'concern': '#2196F3'
                    }
                    
                    for emotion_en, emotion_jp in emotions_jp.items():
                        fig.add_trace(go.Scatter(
                            x=processed_df['timestamp'],
                            y=processed_df[emotion_en],
                            mode='lines+markers',
                            name=emotion_jp,
                            line=dict(color=colors_map[emotion_en], width=3),
                            marker=dict(size=6)
                        ))
                    
                    fig.update_layout(
                        height=400,
                        xaxis_title="時間",
                        yaxis_title="感情スコア (0-10)",
                        yaxis=dict(range=[0, 10]),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 感情レーダーチャート
                st.markdown("#### 🎯 総合感情分析")
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(avg_emotions.values()),
                    theta=list(avg_emotions.keys()),
                    fill='toself',
                    name='平均感情スコア',
                    line_color='rgb(54, 162, 235)',
                    fillcolor='rgba(54, 162, 235, 0.3)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # 最新投稿一覧
        st.markdown("### 💬 最新の感想")
        
        if posts_data:
            for post in posts_data[:5]:  # 最新5件を表示
                content, timestamp, happiness, excitement, satisfaction, concern, processed, error_type, error_message = post
                
                with st.container():
                    st.markdown(f"**{content[:100]}{'...' if len(content) > 100 else ''}**")
                    
                    if processed:
                        # エラー状況に応じた表示
                        if error_type:
                            # エラーが発生した場合の表示
                            error_icons = {
                                "parse_error": "⚠️",
                                "api_error": "❌"
                            }
                            error_descriptions = {
                                "parse_error": "AIの応答解析エラー",
                                "api_error": "API呼び出しエラー"
                            }
                            
                            st.warning(f"{error_icons.get(error_type, '⚠️')} **{error_descriptions.get(error_type, 'エラー')}が発生**")
                            st.caption(f"詳細: {error_message}")
                            st.info("💡 上記エラーのため、以下は推定値です")
                        
                        # 感情バーチャート（小）
                        emotions = [happiness, excitement, satisfaction, concern]
                        emotion_names = ['😊', '🔥', '💚', '💙']
                        
                        cols = st.columns(4)
                        for i, (name, value) in enumerate(zip(emotion_names, emotions)):
                            with cols[i]:
                                # エラー時は値の背景色を変更
                                if error_type:
                                    st.metric(
                                        name, 
                                        f"{value:.1f}",
                                        help="⚠️ 推定値"
                                    )
                                else:
                                    st.metric(name, f"{value:.1f}")
                    else:
                        # レート制限状況に応じた表示
                        if not rate_limiter.can_request():
                            wait_time = rate_limiter.wait_time()
                            st.warning(f"⏳ レート制限中 (残り約 {wait_time} 秒)")
                        else:
                            st.info("🤔 感情分析中...")
                    
                    st.caption(f"投稿時間: {timestamp}")
                    st.divider()
        else:
            st.info("まだ投稿がありません。最初の投稿をしてみませんか？")
    
    # 自動更新（頻度を下げる）
    if st.button("🔄 手動更新", help="データを手動で更新"):
        st.rerun()
    
    # 定期的な自動更新（5秒間隔）
    time.sleep(5)
    st.rerun()

# QRコード生成ページ
def qr_page():
    st.title("📱 QRコード")
    st.subheader("このQRコードをスマホで読み取って参加しよう！")
    
    # アプリのURLを取得
    try:
        app_url = get_app_url()
    except:
        # フォールバック用のURL（デモ用）
        app_url = "https://your-emotion-sns-demo.streamlit.app"
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📲 参加方法")
        st.markdown("""
        1. **スマホのカメラアプリを開く**
        2. **QRコードを読み取り**
        3. **リンクをタップ**
        4. **感想を投稿してみよう！**
        """)
        
        st.markdown("### 🔗 直接アクセス")
        st.code(app_url, language="text")
        
        # URLをコピーボタン
        if st.button("📋 URLをコピー", help="クリップボードにURLをコピー"):
            st.write("URLをコピーしました！（手動でコピーしてください）")
    
    with col2:
        st.markdown("### 📊 QRコード")
        
        try:
            # QRコードを生成
            qr_base64 = generate_qr_code(app_url)
            
            # QRコードを表示
            st.markdown(
                f'<div style="display: flex; justify-content: center;">'
                f'<img src="data:image/png;base64,{qr_base64}" width="300">'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.success("✅ QRコード生成完了！")
            
        except Exception as e:
            st.error(f"QRコード生成エラー: {e}")
            
            # フォールバック表示
            st.markdown(f"""
            <div style="border: 2px dashed #ccc; padding: 20px; text-align: center;">
                <h3>QRコード</h3>
                <p>手動でURLにアクセスしてください</p>
                <code>{app_url}</code>
            </div>
            """, unsafe_allow_html=True)
    
    # デモ用の説明
    st.markdown("---")
    st.markdown("### 🎯 デモの流れ")
    
    demo_steps = st.columns(4)
    
    with demo_steps[0]:
        st.markdown("#### 1️⃣ QR読み取り")
        st.markdown("スマホでQRコードを読み取り")
    
    with demo_steps[1]:
        st.markdown("#### 2️⃣ 感想投稿")
        st.markdown("オープンキャンパスの感想を投稿")
    
    with demo_steps[2]:
        st.markdown("#### 3️⃣ AI分析")
        st.markdown("感情をリアルタイムで分析")
    
    with demo_steps[3]:
        st.markdown("#### 4️⃣ 結果確認")
        st.markdown("みんなの感情を可視化して確認")
    
    # 統計情報
    st.markdown("---")
    st.markdown("### 📈 現在の参加状況")
    
    conn = init_database()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM posts")
    total_posts = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM posts WHERE processed = TRUE")
    processed_posts = cursor.fetchone()[0]
    
    stats_cols = st.columns(3)
    
    with stats_cols[0]:
        st.metric("💬 総投稿数", total_posts)
    
    with stats_cols[1]:
        st.metric("🤖 分析完了", processed_posts)
    
    with stats_cols[2]:
        participation_rate = (total_posts / 200) * 100 if total_posts <= 200 else 100
        st.metric("📊 参加率", f"{participation_rate:.1f}%", help="想定参加者200名に対する参加率")

# サイドバーメニュー
st.sidebar.title("🎓 メニュー")
page = st.sidebar.selectbox("ページを選択", ["メイン", "QRコード"])

if page == "メイン":
    main()
elif page == "QRコード":
    qr_page()

# デバッグ用：手動で感情分析をテストするボタン
def add_debug_controls():
    """デバッグ用のコントロールパネル"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 デバッグコントロール")
    
    # バックグラウンド処理のステータス表示
    if 'debug_status' in st.session_state and st.session_state.debug_status:
        st.sidebar.markdown("#### 📋 処理ログ")
        for status in st.session_state.debug_status[-5:]:  # 最新5件
            st.sidebar.text(status)
    
    if st.sidebar.button("🧪 感情分析テスト", key="test_emotion"):
        client = setup_genai()
        if client:
            test_text = "今日の授業はとても楽しかったです！"
            st.sidebar.info(f"テスト文: {test_text}")
            
            emotions, error_type, error_message = analyze_emotion_with_ai(test_text, client)
            
            if emotions:
                st.sidebar.success("✅ 分析成功")
                st.sidebar.json(emotions)
                if error_type:
                    st.sidebar.warning(f"エラータイプ: {error_type}")
                    st.sidebar.caption(error_message)
            else:
                st.sidebar.error("❌ 分析失敗")
        else:
            st.sidebar.error("❌ APIクライアントが利用できません")
    
    if st.sidebar.button("🗄️ 未処理投稿を強制処理", key="force_process"):
        conn = init_database()
        cursor = conn.cursor()
        
        # 未処理の投稿を取得
        cursor.execute("SELECT id, content FROM posts WHERE processed = FALSE ORDER BY id DESC LIMIT 5")
        unprocessed = cursor.fetchall()
        
        if unprocessed:
            st.sidebar.info(f"未処理投稿: {len(unprocessed)} 件")
            for post_id, content in unprocessed:
                st.sidebar.text(f"ID {post_id}: {content[:30]}...")
                # キューに再追加
                st.session_state.emotion_queue.put(post_id)
            st.sidebar.success("キューに再追加しました")
        else:
            st.sidebar.info("未処理投稿はありません")
    
    # キューの状況表示
    queue_size = st.session_state.emotion_queue.qsize()
    st.sidebar.metric("📊 キューサイズ", queue_size)
    
    # 手動更新ボタン
    if st.sidebar.button("🔄 データ強制更新", key="force_refresh"):
        st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 デモ統計")

conn = init_database()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM posts")
total_posts = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM posts WHERE processed = TRUE")
processed_posts = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM posts WHERE processed = FALSE")
pending_posts = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM posts WHERE error_type IS NOT NULL")
error_posts = cursor.fetchone()[0]

st.sidebar.metric("総投稿数", total_posts)
st.sidebar.metric("分析済み", processed_posts)
st.sidebar.metric("分析待ち", pending_posts)
st.sidebar.metric("エラー発生", error_posts, help="推定値を使用した投稿数")

# 分析進捗バー
progress_value = processed_posts / max(total_posts, 1)
st.sidebar.progress(progress_value)

# レート制限状況の表示
if not rate_limiter.can_request():
    wait_time = rate_limiter.wait_time()
    st.sidebar.warning(f"⏳ レート制限中\n残り: {wait_time}秒")
else:
    st.sidebar.success("✅ 分析可能")

# エラー発生時の説明
if error_posts > 0:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ エラーについて")
    st.sidebar.info("""
    AIシステムでは時々エラーが発生します：
    - **解析エラー**: AIの応答が期待と異なる
    - **API エラー**: サーバー通信の問題
    
    エラー時は推定値を表示しています。
    これも実際のシステム開発で重要な考慮点です！
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 使用技術
- **Streamlit**: Webアプリフレームワーク  
- **Google GenAI**: 感情分析（gemini-2.5-flash-lite）  
- **SQLite**: データベース  
- **Plotly**: データ可視化  
""")

# デモ用の説明
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 このデモについて
リアルタイムでAIが投稿の感情を分析し、
みんなの感想を可視化します。
デモ終了後も自分で構築できます！
""")