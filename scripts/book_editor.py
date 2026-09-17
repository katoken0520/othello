import os
import json
import sys
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine
from othello_env import config as cf
from othello_env import utils

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- ユーザーに評価値を尋ねるカスタムダイアログ ---
class ScoreInputDialog(ctk.CTkToplevel):
    def __init__(self, parent, default_value="0.0"):
        super().__init__(parent)
        self.title("定石評価値の登録")
        self.geometry("450x260")
        self.resizable(False, False)
        self.transient(parent) # 親ウィンドウの前に表示
        self.grab_set()        # このダイアログ以外を操作不能にする

        self.result = None

        # 誤解を防ぐための丁寧な説明文
        desc = (
            "この手に対する評価値を入力・修正してください。\n\n"
            "【 ⚠ 評価値の基準について 】\n"
            "ここは「自分目線（相対評価）」での登録になります。\n"
            "・現在の手番にとって有利な手なら「プラス (+)」\n"
            "・現在の手番にとって不利な手なら「マイナス (-)」\n"
        )
        self.lbl = ctk.CTkLabel(self, text=desc, font=("Meiryo", 13), justify="left", text_color="#e2e8f0")
        self.lbl.pack(pady=(20, 10), padx=20)

        # 入力ボックス（AIの評価値をデフォルトでセット）
        self.entry = ctk.CTkEntry(self, width=200, font=("Consolas", 16, "bold"), justify="center")
        self.entry.insert(0, str(default_value))
        self.entry.pack(pady=10)
        self.entry.focus_set()

        # ボタン類
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(10, 20))

        self.ok_btn = ctk.CTkButton(btn_frame, text="登録する", command=self.on_ok, width=120, font=("Meiryo", 13, "bold"))
        self.ok_btn.pack(side="left", padx=10)

        self.cancel_btn = ctk.CTkButton(btn_frame, text="キャンセル", command=self.on_cancel, width=120, fg_color="#718096", hover_color="#4a5568", font=("Meiryo", 13))
        self.cancel_btn.pack(side="right", padx=10)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.on_cancel())

    def on_ok(self):
        try:
            self.result = float(self.entry.get())
            self.destroy()
        except ValueError:
            messagebox.showerror("入力エラー", "有効な数値を入力してください", parent=self)

    def on_cancel(self):
        self.destroy()


class BookEditorGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Othello Opening Book Editor")
        self.window.geometry("920x660")
        self.window.resizable(False, False)
        self.window.configure(fg_color="#1a1a1a")

        self.book_data = self.load_book()
        self.reset_state()
        
        self.ai_engine = engine.SearchEngine()
        self.init_ai_engine()

        self.current_thread_id = 0
        self.engine_lock = threading.Lock()
        self.eval_scores = {}

        self.build_ui()
        self.update_gui()
        self.start_eval_mode() 

    def load_book(self):
        if os.path.exists(cf.BOOK_PATH):
            with open(cf.BOOK_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_book(self):
        with open(cf.BOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(self.book_data, f, indent=4)

    def reset_state(self):
        self.board = ((1 << 28) | (1 << 35), (1 << 27) | (1 << 36))
        self.order = 1
        self.history = []

    def init_ai_engine(self):
        self.ai_engine.clear_book()
        latest_model_gen = utils.get_latest_model_gen() if hasattr(utils, 'get_latest_model_gen') else 0
        if latest_model_gen > 0:
            model_dir = os.path.join(cf.BASE_DIR, "data", "learned", f"gen{latest_model_gen}")
            structure_path = os.path.join(model_dir, "network_structure.json")
            cpp_weights_path = os.path.join(model_dir, "cpp_weights.bin")
            try:
                with open(structure_path, "r", encoding="utf-8") as f:
                    struct = json.load(f)
                pattern_bits = [feature['bits'] for feature in struct["NET_STLC"]]
                self.ai_engine.set_structure(pattern_bits, struct["MLP_LAYERS"], struct["ACTIVE_FEATURES"])
                self.ai_engine.load_weights(cpp_weights_path)
            except Exception as e:
                print(f"[ERROR] モデルロード失敗: {e}")
        else:
            pattern_bits = [feature['bits'] for feature in cf.NET_STLC]
            self.ai_engine.set_structure(pattern_bits, cf.MLP_LAYERS, cf.ACTIVE_FEATURES)

        # JSONから定石を読み込み、全ての手とスコアを登録する
        book = self.load_book()
        for h, v in book.items():
            for move_data in v:
                self.ai_engine.add_book_move(int(h), int(move_data[0]), float(move_data[1]))

    def build_ui(self):
        self.canvas_size = 560
        self.cell_size = self.canvas_size // 8
        self.canvas = tk.Canvas(self.window, width=self.canvas_size, height=self.canvas_size, 
                                bg="#2c5282", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=25, pady=25)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.ctrl_frame = ctk.CTkFrame(self.window, fg_color="#262626", corner_radius=12)
        self.ctrl_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 25), pady=25)
        self.ctrl_frame.grid_propagate(False)
        self.ctrl_frame.configure(width=280)

        info_inner = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        info_inner.pack(fill="x", padx=18, pady=(20, 10))
        self.turn_label = ctk.CTkLabel(info_inner, text="手番: 黒", font=("Meiryo", 16, "bold"), text_color="#ffffff")
        self.turn_label.pack(anchor="w")
        self.hash_label = ctk.CTkLabel(info_inner, text="Hash: -", font=("Consolas", 11), text_color="#aaaaaa")
        self.hash_label.pack(anchor="w")

        mode_panel = ctk.CTkFrame(self.ctrl_frame, fg_color="#333333", corner_radius=8)
        mode_panel.pack(fill="x", padx=18, pady=10)
        
        self.register_mode_var = ctk.BooleanVar(value=False)
        self.mode_switch = ctk.CTkSwitch(mode_panel, text="盤面クリックで定石に登録", 
                                         variable=self.register_mode_var, font=("Meiryo", 12, "bold"),
                                         progress_color="#d69e2e")
        self.mode_switch.pack(pady=(12, 5), padx=10, anchor="w")

        self.symmetric_register_var = ctk.BooleanVar(value=True)
        self.sym_switch = ctk.CTkSwitch(mode_panel, text="対称形(回転/反転)も自動登録", 
                                         variable=self.symmetric_register_var, font=("Meiryo", 11))
        self.sym_switch.pack(pady=(5, 12), padx=10, anchor="w")

        btn_inner = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        btn_inner.pack(fill="x", padx=18, pady=10)
        ctk.CTkButton(btn_inner, text="⟲ 1手戻す", font=("Meiryo", 13), command=self.undo_move, width=115, fg_color="#4a4a4a").pack(side="left")
        ctk.CTkButton(btn_inner, text="↻ 初期配置", font=("Meiryo", 13), command=self.reset_game, width=115, fg_color="#4a4a4a").pack(side="right")

        ctk.CTkButton(self.ctrl_frame, text="✖ 現局面の定石を削除", font=("Meiryo", 13), 
                      command=self.delete_current_book, fg_color="#9b2c2c", hover_color="#c53030").pack(fill="x", padx=18, pady=10)

        ctk.CTkFrame(self.ctrl_frame, height=2, fg_color="#3a3a3a").pack(fill="x", padx=15, pady=8)
        ctk.CTkLabel(self.ctrl_frame, text="AIの分析深さ (参考用):", font=("Meiryo", 12)).pack(anchor="w", padx=18)
        self.depth_var = tk.IntVar(value=cf.DEPTH)
        self.depth_slider = ctk.CTkSlider(self.ctrl_frame, from_=2, to=16, number_of_steps=14, 
                                          variable=self.depth_var, command=self.update_depth_text)
        self.depth_slider.pack(fill="x", padx=18, pady=(0, 5))
        self.depth_lbl = ctk.CTkLabel(self.ctrl_frame, text=str(cf.DEPTH), font=("Meiryo", 12, "bold"))
        self.depth_lbl.pack(anchor="e", padx=18)

        self.status_label = ctk.CTkLabel(self.ctrl_frame, text="準備完了", font=("Meiryo", 11), text_color="#718096")
        self.status_label.pack(side="bottom", anchor="w", padx=15, pady=12)

    def update_depth_text(self, val):
        self.depth_lbl.configure(text=str(int(val)))
        if hasattr(self, '_slider_timer'):
            self.window.after_cancel(self._slider_timer)
        self._slider_timer = self.window.after(300, self.start_eval_mode)

    def get_current_hash_str(self):
        p_bb = self.board[0] if self.order == 1 else self.board[1]
        o_bb = self.board[1] if self.order == 1 else self.board[0]
        return str(self.ai_engine.get_board_hash(p_bb, o_bb))

    # --- 盤面の対称形を生成するロジック ---
    def transform_board(self, p_bb, o_bb, idx, rot, flip):
        """盤面を90度回転(rot回)と左右反転(flip)して新しい状態を返す"""
        p_arr = [[(p_bb >> (r * 8 + c)) & 1 for c in range(8)] for r in range(8)]
        o_arr = [[(o_bb >> (r * 8 + c)) & 1 for c in range(8)] for r in range(8)]
        ir, ic = idx // 8, idx % 8

        if flip:
            p_arr = [row[::-1] for row in p_arr]
            o_arr = [row[::-1] for row in o_arr]
            ic = 7 - ic

        for _ in range(rot):
            p_arr = [list(x) for x in zip(*p_arr[::-1])]
            o_arr = [list(x) for x in zip(*o_arr[::-1])]
            ir, ic = ic, 7 - ir
            
        new_p = 0
        new_o = 0
        for r in range(8):
            for c in range(8):
                if p_arr[r][c]: new_p |= (1 << (r * 8 + c))
                if o_arr[r][c]: new_o |= (1 << (r * 8 + c))
        return new_p, new_o, ir * 8 + ic

    def get_symmetries(self, p_bb, o_bb, idx):
        """回転と反転を組み合わせて、全8パターンの対称形をSetで返す"""
        syms = set()
        for rot in range(4):
            for flip in [False, True]:
                syms.add(self.transform_board(p_bb, o_bb, idx, rot, flip))
        return syms

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(1, 8):
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.canvas_size, fill="#1a365d", width=2)
            self.canvas.create_line(0, i * self.cell_size, self.canvas_size, i * self.cell_size, fill="#1a365d", width=2)
            
        for r, c in [(2,2), (2,6), (6,2), (6,6)]:
            x, y = c * self.cell_size, r * self.cell_size
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="#1a365d", outline="")

        legal_moves = utils.get_valid_moves(self.board, self.order)
        legal_list = list(utils.iter_set_bits(legal_moves))
        
        current_hash = self.get_current_hash_str()
        
        # この盤面に登録されている全ての手(idx)を取得
        book_moves = [m[0] for m in self.book_data.get(current_hash, [])]

        for idx in range(64):
            r, c = idx // 8, idx % 8
            cx, cy = c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2
            radius = self.cell_size * 0.43
            
            b, w = (self.board[0] >> idx) & 1, (self.board[1] >> idx) & 1
            
            if b == 1:
                self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, fill="#151515", outline="#000000", width=1)
            elif w == 1:
                self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, fill="#f3f3f3", outline="#b5b5b5", width=1)
            elif idx in legal_list:
                # 定石に登録されている手はゴールドリングで強調表示
                if idx in book_moves:
                    self.canvas.create_oval(cx-radius*0.8, cy-radius*0.8, cx+radius*0.8, cy+radius*0.8, outline="#d69e2e", width=4)
                else:
                    self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill="#4299e1", outline="")
                
                if idx in self.eval_scores:
                    score = self.eval_scores[idx]
                    rounded_score = max(-64, min(64, int(round(score, 0))))
                    txt = f"+{rounded_score}" if rounded_score > 0 else str(rounded_score)
                    color = "#ffffff" if idx not in book_moves else "#ecc94b"
                    
                    self.canvas.create_text(cx+1, cy+1, text=txt, fill="#000000", font=("Impact", 13))
                    self.canvas.create_text(cx, cy, text=txt, fill=color, font=("Impact", 13))

    def update_gui(self):
        self.draw_board()
        turn_str = "黒" if self.order == 1 else "白" if self.order == -1 else "終了"
        self.turn_label.configure(text=f"手番: {turn_str}")
        self.hash_label.configure(text=f"Hash: {self.get_current_hash_str()[-8:]}...")

    def ask_for_score(self, default_score):
        """ユーザーにダイアログでスコアを入力させるヘルパー"""
        dialog = ScoreInputDialog(self.window, default_value=round(default_score, 2))
        self.window.wait_window(dialog)
        return dialog.result

    def on_canvas_click(self, event):
        if self.order == 0: return

        c, r = event.x // self.cell_size, event.y // self.cell_size
        idx = r * 8 + c
        
        legal = utils.get_valid_moves(self.board, self.order)
        if (legal >> idx) & 1:
            if self.register_mode_var.get():
                ai_score = self.eval_scores.get(idx, 0.0) 
                
                # ダイアログでユーザーに確認・変更させる
                user_score = self.ask_for_score(ai_score)
                if user_score is None:
                    return # キャンセルされた場合は打たずに中止

                p_bb = self.board[0] if self.order == 1 else self.board[1]
                o_bb = self.board[1] if self.order == 1 else self.board[0]

                # 対称形の生成
                if self.symmetric_register_var.get():
                    syms = self.get_symmetries(p_bb, o_bb, idx)
                else:
                    syms = {(p_bb, o_bb, idx)}

                count = 0
                for tp, to, ti in syms:
                    h_str = str(self.ai_engine.get_board_hash(tp, to))
                    if h_str not in self.book_data:
                        self.book_data[h_str] = []
                    
                    # すでに同じ手が登録されていなければ追加、あればスコアだけ更新
                    found = False
                    for m in self.book_data[h_str]:
                        if m[0] == ti:
                            m[1] = user_score
                            found = True
                            break
                    if not found:
                        self.book_data[h_str].append([ti, user_score])
                    count += 1

                self.save_book()
                self.init_ai_engine() # エンジン側のメモリも更新
                self.status_label.configure(text=f"★ 定石を登録しました (対称形含め {count} パターン)")

            self.execute_move(idx)

    def execute_move(self, idx):
        self.history.append((self.board, self.order))
        
        self.current_thread_id += 1
        self.ai_engine.stop()
        self.eval_scores.clear()

        self.board, finish_flag, pass_flag = utils.put_stone(self.board, idx, self.order)
        
        if not pass_flag:
            self.order *= -1
            
        if finish_flag != 'CONTINUE':
            self.order = 0
            self.update_gui()
            messagebox.showinfo("終局", "ゲーム終了です。")
            return

        self.update_gui()
        self.start_eval_mode() 

    def delete_current_book(self):
        # 現在の相対盤面を取得
        p_bb = self.board[0] if self.order == 1 else self.board[1]
        o_bb = self.board[1] if self.order == 1 else self.board[0]

        # 登録時と同じく対称形スイッチを判定
        # (削除時はどの「手」かは関係ないため、idxにはダミーの0を渡します)
        if self.symmetric_register_var.get():
            syms = self.get_symmetries(p_bb, o_bb, 0)
        else:
            syms = {(p_bb, o_bb, 0)}

        count = 0
        for tp, to, _ in syms:
            h_str = str(self.ai_engine.get_board_hash(tp, to))
            if h_str in self.book_data:
                del self.book_data[h_str]
                count += 1

        if count > 0:
            self.save_book()
            self.init_ai_engine() # エンジン側のメモリも更新
            self.status_label.configure(text=f"✖ 現局面の定石を削除しました (対称形含め {count} パターン)")
            self.update_gui()
        else:
            self.status_label.configure(text="この局面には定石が登録されていません")

    def undo_move(self):
        if not self.history: return
        self.current_thread_id += 1
        self.ai_engine.stop()
        self.eval_scores.clear()
        
        self.board, self.order = self.history.pop()
        self.update_gui()
        self.start_eval_mode()

    def reset_game(self):
        self.current_thread_id += 1
        self.ai_engine.stop()
        self.eval_scores.clear()
        self.reset_state()
        self.update_gui()
        self.start_eval_mode()

    def start_eval_mode(self):
        if self.order == 0: return
        self.current_thread_id += 1
        self.eval_scores.clear()
        self.status_label.configure(text="局面を分析中...")
        self.window.update_idletasks()
        
        thread = threading.Thread(target=self._eval_thread_worker, args=(self.current_thread_id, self.board, self.order), daemon=True)
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
        self.status_label.configure(text=f"分析完了 (深さ: {now_depth})")
        self.update_gui()

if __name__ == "__main__":
    window = ctk.CTk()
    app = BookEditorGUI(window)
    window.mainloop()