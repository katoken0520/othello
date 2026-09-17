import os
import json
import sys
import random
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine
from othello_env import config as cf
from othello_env import utils

# CustomTkinterの全体テーマ設定
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AdvancedOthelloGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Othello AI Analyzer & Game")
        self.window.geometry("900x660") # メニュー追加に伴い縦幅を少し拡張
        self.window.resizable(False, False)
        self.window.configure(fg_color="#1a1a1a")

        # --- 状態管理変数 ---
        self.reset_state()
        
        # AIエンジンの初期化 (起動時は自動的に最新世代をロード)
        self.init_ai_engine()

        # スレッド・分析管理用
        self.current_thread_id = 0
        self.engine_lock = threading.Lock()
        self.eval_scores = {}      # 各合法手の評価値 {idx: score}
        self.latest_ai_value = 0   # AIが最後に算出した全体の評価値
        self.hide_legal_dots = False # パスダイアログ表示時などにドットを隠すフラグ

        # --- UIレイアウト構築 ---
        self.build_ui()
        self.update_gui()

    def reset_state(self):
        """ゲーム状態の完全初期化"""
        self.board = ((1 << 28) | (1 << 35), (1 << 27) | (1 << 36)) # (初期黒, 初期白)
        self.order = 1 # 1:黒, -1:白
        self.history = [] # 巻き戻し(Undo)用履歴スタック [(board, order), ...]
        
        # モード管理 ('human': 人vs人, 'match': AI対戦, 'eval': 局面分析)
        self.mode = 'human'
        self.ai_color = 0 # 1:AIが黒, -1:AIが白
        self.ai_is_thinking = False

    def get_available_models(self):
        """data/learned/ フォルダ内を探索し、有効なAI世代のリストを返す"""
        import glob
        import re
        dirs = glob.glob(os.path.join(cf.BASE_DIR, "data", "learned", "gen*"))
        gens = []
        for d in dirs:
            match = re.search(r'gen(\d+)', os.path.basename(d))
            if match and os.path.exists(os.path.join(d, "cpp_weights.bin")):
                gens.append(int(match.group(1)))
        gens.sort()
        return gens

    def init_ai_engine(self, gen_num=None):
        """指定された、または最新のAIモデル世代をロードして初期化（古い重みを確実に消すためインスタンスを再生成）"""
        self.ai_engine = engine.SearchEngine()
        
        if gen_num is None:
            latest_model_gen = utils.get_latest_model_gen() if hasattr(utils, 'get_latest_model_gen') else 0
            gen_num = latest_model_gen

        if gen_num > 0:
            model_dir = os.path.join(cf.BASE_DIR, "data", "learned", f"gen{gen_num}")
            structure_path = os.path.join(model_dir, "network_structure.json")
            cpp_weights_path = os.path.join(model_dir, "cpp_weights.bin")
            try:
                with open(structure_path, "r", encoding="utf-8") as f:
                    struct = json.load(f)
                pattern_bits = [feature['bits'] for feature in struct["NET_STLC"]]
                self.ai_engine.set_structure(pattern_bits, struct["MLP_LAYERS"], struct["ACTIVE_FEATURES"])
                self.ai_engine.load_weights(cpp_weights_path)
                print(f"[SYSTEM] モデル [gen{gen_num}] をロードしました。")
                if hasattr(self, 'status_label'):
                    self.status_label.configure(text=f"モデル [gen{gen_num}] ロード完了")
            except Exception as e:
                print(f"[ERROR] モデルのロードに失敗しました: {e}")
                if hasattr(self, 'status_label'):
                    self.status_label.configure(text="モデルロード失敗")
        else:
            pattern_bits = [feature['bits'] for feature in cf.NET_STLC]
            self.ai_engine.set_structure(pattern_bits, cf.MLP_LAYERS, cf.ACTIVE_FEATURES)
            print("[SYSTEM] デフォルトの石数評価で動作します。")
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="デフォルト (石数評価) ロード完了")

        # 複数手・スコア対応の定石ブックを一括登録
        if os.path.exists(cf.BOOK_PATH):
            with open(cf.BOOK_PATH, "r", encoding="utf-8") as f:
                book = json.load(f)
                for hash_str, move_data in book.items():
                    for move, score in move_data:
                        self.ai_engine.add_book_move(int(hash_str), int(move), float(score))

    def on_model_selected(self, choice):
        """ユーザーがドロップダウンからモデルを変更した際のイベントハンドラ"""
        self.current_thread_id += 1
        self.ai_engine.stop() # 現在進行中の計算を緊急停止
        self.eval_scores.clear()
        
        import re
        match = re.search(r'gen(\d+)', choice)
        if match:
            gen_num = int(match.group(1))
            self.init_ai_engine(gen_num)
        else:
            self.init_ai_engine(0) # 0番はデフォルト(石数評価)
            
        # 分析モード中だった場合は、新しい脳細胞で即座に再計算を開始
        if self.mode == 'eval':
            self.start_eval_mode()
        else:
            self.update_gui()

    def build_ui(self):
        """洗練されたダークモードUIの構築"""
        # 左側: 盤面キャンバス
        self.canvas_size = 560
        self.cell_size = self.canvas_size // 8
        self.canvas = tk.Canvas(self.window, width=self.canvas_size, height=self.canvas_size, 
                                bg="#1e5631", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=25, pady=25)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # 右側: コントロールコンテナ
        self.ctrl_frame = ctk.CTkFrame(self.window, fg_color="#262626", corner_radius=12)
        self.ctrl_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 25), pady=25)
        self.ctrl_frame.grid_propagate(False)
        self.ctrl_frame.configure(width=265)

        # -- ステータス・情報表示 --
        info_inner = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        info_inner.pack(fill="x", padx=18, pady=(20, 10))
        
        self.turn_label = ctk.CTkLabel(info_inner, text="手番: 黒", font=("Meiryo", 16, "bold"), text_color="#ffffff")
        self.turn_label.pack(anchor="w")
        
        self.score_label = ctk.CTkLabel(info_inner, text="黒: 2   白: 2", font=("Meiryo", 14), text_color="#cccccc")
        self.score_label.pack(anchor="w", pady=(4, 0))
        
        self.value_label = ctk.CTkLabel(info_inner, text="評価値: 0", font=("Meiryo", 14), text_color="#3182ce")
        self.value_label.pack(anchor="w", pady=(4, 0))

        # -- 基本操作 (戻す / リセット) --
        btn_inner = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        btn_inner.pack(fill="x", padx=18, pady=10)
        
        self.undo_btn = ctk.CTkButton(btn_inner, text="⟲ 1手戻す", font=("Meiryo", 14), command=self.undo_move, width=110, hover_color="#404040", fg_color="#4a4a4a")
        self.undo_btn.pack(side="left")
        
        self.reset_btn = ctk.CTkButton(btn_inner, text="↻ リセット", font=("Meiryo", 14), command=self.reset_game, width=110, hover_color="#9b2c2c", fg_color="#9b2c2c")
        self.reset_btn.pack(side="right")

        # 区切り線
        ctk.CTkFrame(self.ctrl_frame, height=2, fg_color="#3a3a3a").pack(fill="x", padx=15, pady=8)

        # -- ★新設: AIモデル選択パネル --
        model_panel = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        model_panel.pack(fill="x", padx=18, pady=4)
        ctk.CTkLabel(model_panel, text="【 使用AIモデルの選択 】", font=("Meiryo", 13, "bold"), text_color="#a0aec0").pack(anchor="w")
        
        available_gens = self.get_available_models()
        self.model_options = ["デフォルト (石数評価)"] + [f"gen{g}" for g in available_gens]
        
        latest_model_gen = utils.get_latest_model_gen() if hasattr(utils, 'get_latest_model_gen') else 0
        default_choice = f"gen{latest_model_gen}" if latest_model_gen > 0 else "デフォルト (石数評価)"
        
        self.model_menu = ctk.CTkOptionMenu(model_panel, values=self.model_options, command=self.on_model_selected, font=("Meiryo", 12))
        self.model_menu.set(default_choice)
        self.model_menu.pack(fill="x", pady=(6, 0))

        # 区切り線
        ctk.CTkFrame(self.ctrl_frame, height=2, fg_color="#3a3a3a").pack(fill="x", padx=15, pady=8)

        # -- 局面分析モードパネル --
        eval_panel = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        eval_panel.pack(fill="x", padx=18, pady=4)
        ctk.CTkLabel(eval_panel, text="【 局面分析機能 】", font=("Meiryo", 13, "bold"), text_color="#a0aec0").pack(anchor="w")
        self.eval_btn = ctk.CTkButton(eval_panel, text="分析を開始", font=("Meiryo", 14), command=self.start_eval_mode, fg_color="#2c7a7b", hover_color="#234e52")
        self.eval_btn.pack(fill="x", pady=(6, 0))

        # 区切り線
        ctk.CTkFrame(self.ctrl_frame, height=2, fg_color="#3a3a3a").pack(fill="x", padx=15, pady=8)

        # -- AI対戦設定パネル --
        match_panel = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        match_panel.pack(fill="x", padx=18, pady=4)
        ctk.CTkLabel(match_panel, text="【 現局面からAI対戦 】", font=("Meiryo", 13, "bold"), text_color="#a0aec0").pack(anchor="w")
        
        self.ai_side_var = tk.IntVar(value=-1)
        self.radio_black = ctk.CTkRadioButton(match_panel, text="AIに黒番(先手)を持たせる", variable=self.ai_side_var, value=1, font=("Meiryo", 12))
        self.radio_black.pack(anchor="w", pady=3)
        self.radio_white = ctk.CTkRadioButton(match_panel, text="AIに白番(後手)を持たせる", variable=self.ai_side_var, value=-1, font=("Meiryo", 12))
        self.radio_white.pack(anchor="w", pady=3)

        ctk.CTkLabel(match_panel, text="AIの思考深さ:", font=("Meiryo", 12)).pack(anchor="w", pady=(6, 0))
        slider_row = ctk.CTkFrame(match_panel, fg_color="transparent")
        slider_row.pack(fill="x")
        self.depth_var = tk.IntVar(value=cf.DEPTH)
        self.depth_slider = ctk.CTkSlider(slider_row, from_=2, to=16, number_of_steps=14, variable=self.depth_var, command=self.update_depth_text, width=170)
        self.depth_slider.pack(side="left", pady=5)
        self.depth_lbl = ctk.CTkLabel(slider_row, text=str(cf.DEPTH), font=("Meiryo", 12, "bold"))
        self.depth_lbl.pack(side="right", padx=(5, 0))

        self.show_eval_match_var = ctk.BooleanVar(value=True)
        self.chk_eval = ctk.CTkCheckBox(match_panel, text="対戦中も盤面に評価値を表示", variable=self.show_eval_match_var, font=("Meiryo", 11))
        self.chk_eval.pack(anchor="w", pady=(6, 12))

        self.match_btn = ctk.CTkButton(match_panel, text="▶ AI対戦をスタート", font=("Meiryo", 14), command=self.start_ai_match, fg_color="#2b6cb0", hover_color="#2b4c7e")
        self.match_btn.pack(fill="x")

        # フッター通知メッセージ
        self.status_label = ctk.CTkLabel(self.ctrl_frame, text="システム準備完了", font=("Meiryo", 11), text_color="#718096")
        self.status_label.pack(side="bottom", anchor="w", padx=15, pady=12)

    def update_depth_text(self, val):
        self.depth_lbl.configure(text=str(int(val)))
        if self.mode == 'eval': 
            if hasattr(self, '_slider_timer'):
                self.window.after_cancel(self._slider_timer)
            self._slider_timer = self.window.after(300, self.start_eval_mode)

    # --- 盤面グラフィック描画 ---
    def draw_board(self):
        self.canvas.delete("all")
        for i in range(1, 8):
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.canvas_size, fill="#143d22", width=2)
            self.canvas.create_line(0, i * self.cell_size, self.canvas_size, i * self.cell_size, fill="#143d22", width=2)
            
        for r, c in [(2,2), (2,6), (6,2), (6,6)]:
            x, y = c * self.cell_size, r * self.cell_size
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="#143d22", outline="")

        legal_moves = utils.get_valid_moves(self.board, self.order)
        legal_list = list(utils.iter_set_bits(legal_moves))

        for idx in range(64):
            r, c = idx // 8, idx % 8
            cx, cy = c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2
            radius = self.cell_size * 0.43
            
            b, w = (self.board[0] >> idx) & 1, (self.board[1] >> idx) & 1
            
            if b == 1:
                self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, fill="#151515", outline="#000000", width=1)
            elif w == 1:
                self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, fill="#f3f3f3", outline="#b5b5b5", width=1)
            
            elif idx in legal_list and not self.hide_legal_dots:
                has_score = idx in self.eval_scores and (self.mode == 'eval' or (self.mode == 'match' and self.show_eval_match_var.get()))
                
                if not self.ai_is_thinking and not has_score:
                    self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill="#48bb78", outline="")
                
                if has_score:
                    score = self.eval_scores[idx]
                    if abs(score) > 9000:
                        diff = int(abs(score)) - 10000
                        txt = f"黒+{diff}" if score > 0 else f"白+{diff}"
                        color = "#ffffff" if score > 0 else "#cccccc"
                    else:
                        rounded_score = max(-64, min(64, int(round(score, 0))))
                        txt = f"+{rounded_score}" if rounded_score > 0 else str(rounded_score)
                        color = "#ffffff"
                    
                    self.canvas.create_text(cx+1, cy+1, text=txt, fill="#000000", font=("Meiryo", 12, "bold"))
                    self.canvas.create_text(cx, cy, text=txt, fill=color, font=("Meiryo", 12, "bold"))

    def update_gui(self):
        self.draw_board()
        black_num = self.board[0].bit_count()
        white_num = self.board[1].bit_count()
        self.score_label.configure(text=f"黒(あなた): {black_num}個   白(AI): {white_num}個" if self.mode=='match' and self.ai_color==-1 else f"黒: {black_num}個   白: {white_num}個")
        
        if self.mode == 'match':
            mode_str = "（AI対戦中）"
        elif self.mode == 'eval':
            mode_str = "（局面分析中）"
        else:
            mode_str = ""
        turn_str = "黒" if self.order == 1 else "白" if self.order == -1 else "終了"
        self.turn_label.configure(text=f"手番: {turn_str}{mode_str}")

        if abs(self.latest_ai_value) > 9000:
            diff = int(abs(self.latest_ai_value)) - 10000
            if self.latest_ai_value > 0:
                v_text = f"黒勝ち（石差: {abs(diff)}）"
            else:
                v_text = f"白勝ち（石差: {abs(diff)}）"
        else:
            main_v = int(round(self.latest_ai_value, 0))
            main_v = max(-64, min(64, main_v))
            v_text = f"+{main_v}" if main_v > 0 else str(main_v)
            
        self.value_label.configure(text=f"評価値: {v_text}")

    def on_canvas_click(self, event):
        self.current_thread_id += 1
        self.ai_engine.stop()
        self.ai_is_thinking = False
        self.eval_scores.clear()
        
        if self.ai_is_thinking or self.order == 0:
            return

        c, r = event.x // self.cell_size, event.y // self.cell_size
        idx = r * 8 + c
        
        legal = utils.get_valid_moves(self.board, self.order)
        if (legal >> idx) & 1:
            self.execute_move(idx)

    def execute_move(self, idx):
        self.history.append((self.board, self.order))
        self.board, finish_flag, pass_flag = utils.put_stone(self.board, idx, self.order)
        self.eval_scores.clear()
        self.current_thread_id += 1 

        if not pass_flag:
            self.order *= -1
            
        if finish_flag != 'CONTINUE':
            self.order = 0
            self.update_gui()
            self.show_result(finish_flag)
            return

        if pass_flag:
            self.hide_legal_dots = True
            self.update_gui()
            self.window.update_idletasks()
            
            passed_player = "黒" if self.order == -1 else "白"
            messagebox.showinfo("パス発生", f"{passed_player}は配置可能なマスがないためパスとなります。")
            self.hide_legal_dots = False

        self.update_gui()

        if self.mode == 'match' and self.order == self.ai_color:
            self.trigger_ai_move()
        elif self.mode == 'eval':
            self.start_eval_mode()

    def undo_move(self):
        if not self.history: return
        self.current_thread_id += 1
        self.ai_engine.stop()
        self.ai_is_thinking = False
        self.eval_scores.clear()
        
        self.board, self.order = self.history.pop()
        if self.mode == 'match' and self.order == self.ai_color and self.history:
             self.board, self.order = self.history.pop()

        self.update_gui()
        if self.mode == 'eval': self.start_eval_mode()

    def reset_game(self):
        self.current_thread_id += 1
        self.ai_engine.stop()
        self.ai_is_thinking = False
        self.eval_scores.clear()
        self.latest_ai_value = 0
        self.reset_state()
        self.update_gui()
        self.status_label.configure(text="ゲームをリセットしました")

    def show_result(self, finish_flag):
        msg = "引き分けです！" if finish_flag == 'DRAW' else f"{'黒' if finish_flag == 'BLACK' else '白'}の勝利です！"
        messagebox.showinfo("ゲーム終了", msg)
        self.mode = 'human'
        self.status_label.configure(text="対戦終了")

    def start_eval_mode(self):
        if self.order == 0: return
        self.mode = 'eval'
        self.current_thread_id += 1
        self.eval_scores.clear()
        self.window.update_idletasks()
        
        thread = threading.Thread(target=self._eval_thread_worker, args=(self.current_thread_id, self.board, self.order))
        thread.daemon = True
        thread.start()

    def _eval_thread_worker(self, thread_id, board, order):
        with self.engine_lock: 
            if self.current_thread_id != thread_id: return 
            
            p_bb = board[0] if order == 1 else board[1]
            o_bb = board[1] if order == 1 else board[0]
            depth = int(self.depth_var.get())

            empty_num = 64 - (p_bb | o_bb).bit_count()
            if empty_num <= cf.MAIN_FULL_DEPTH:
                 depth = cf.MAIN_FULL_DEPTH

            for d in range(2, depth + 1):
                self.ai_engine.best_move(p_bb, o_bb, d, True)
                if self.current_thread_id != thread_id: return 
                
                root_results = self.ai_engine.get_root_scores()
                scores = {idx: (val if order == 1 else -val) for idx, val in root_results}

                if self.current_thread_id == thread_id:
                    self.window.after(0, self._apply_eval_scores, thread_id, scores, d)

    def _apply_eval_scores(self, thread_id, scores, now_depth):
        if self.current_thread_id != thread_id: return
        self.eval_scores = scores
        if scores:
            self.latest_ai_value = max(scores.values()) if self.order == 1 else min(scores.values())
        self.status_label.configure(text=f"分析結果を表示しました（深さ: {now_depth}）")
        self.update_gui()

    def start_ai_match(self):
        if self.order == 0: return
        self.mode = 'match'
        self.ai_color = self.ai_side_var.get()
        self.eval_scores.clear()
        self.current_thread_id += 1
        
        your_color = "白(後手)" if self.ai_color == 1 else "黒(先手)"
        messagebox.showinfo("対戦モード開始", f"現局面からAI対戦に切り替えます。\nあなたの手番: {your_color}")
        
        self.update_gui()
        if self.order == self.ai_color:
            self.trigger_ai_move()

    def trigger_ai_move(self):
        self.ai_is_thinking = True
        self.status_label.configure(text="AIが思考しています...")
        self.window.update_idletasks()
        
        thread = threading.Thread(target=self._ai_move_worker, args=(self.current_thread_id, self.board, self.order), daemon=True)
        thread.start()

    def _ai_move_worker(self, thread_id, board, order):
        with self.engine_lock:
            if self.current_thread_id != thread_id: return
            
            p_bb = board[0] if order == 1 else board[1]
            o_bb = board[1] if order == 1 else board[0]
            depth = int(self.depth_var.get())
            
            empty_num = 64 - (p_bb | o_bb).bit_count()
            if empty_num <= cf.MAIN_FULL_DEPTH:
                 depth = cf.MAIN_FULL_DEPTH
                 
            idx, value = self.ai_engine.best_move(p_bb, o_bb, depth, False)
            
            if self.current_thread_id == thread_id:
                black_view_value = value if order == 1 else -value
                self.window.after(0, self._apply_ai_move, thread_id, idx, black_view_value)

    def _apply_ai_move(self, thread_id, idx, value):
        if self.current_thread_id != thread_id: return
        self.ai_is_thinking = False
        self.latest_ai_value = value
        self.status_label.configure(text=f"AIが着手しました。")
        self.execute_move(idx)

if __name__ == "__main__":
    window = ctk.CTk()
    app = AdvancedOthelloGUI(window)
    window.mainloop()