import os
import sys
import random
import json
import time
import math

# プロジェクトルートの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine
from othello_env import config as cf
from othello_env import utils

# ==========================================
# ---- テスト設定 ----
GEN_A = 12       # モデルAの世代 (例: 1, 2... / 0を指定するとデフォルトの石数評価)
GEN_B = 11       # モデルBの世代
NUM_CYCLES = 50 # 対戦サイクル数 (1サイクル = 先手・後手入れ替えの2試合。50サイクル=100試合)
RANDOM_MOVES = 20 # 初期盤面をばらけさせるためのランダム手数
DEPTH = 8
FULL_DEPTH = 12
# ---- メモ（常にvsの左側目線） ----
# GEN10 vs GEN9 (DEPTH=8, FULL_DEPTH=12) -> 勝率: 55.5%, 平均: +2.09石, 標準偏差: 7.97石
# GEN11 vs GEN10 (DEPTH=8, FULL_DEPTH=12) -> 勝率: 52.5%, 平均: +0.48石, 標準偏差: 6.32石
# GEN11 vs GEN9 (DEPTH=8, FULL_DEPTH=12) -> 勝率: 49.0%, 平均: +0.02石, 標準偏差: 7.88石
# GEN12 vs GEN11 (DEPTH=8, FULL_DEPTH=12) -> 勝率: 49.0%, 平均: -0.54石, 標準偏差: 7.70石
# ==========================================

def load_engine(gen_num, name):
    """指定された世代のエンジンをロードして返す"""
    ai_engine = engine.SearchEngine()
    
    if gen_num > 0:
        model_dir = os.path.join(cf.BASE_DIR, "data", "learned", f"gen{gen_num}")
        structure_path = os.path.join(model_dir, "network_structure.json")
        cpp_weights_path = os.path.join(model_dir, "cpp_weights.bin")
        
        try:
            with open(structure_path, "r", encoding="utf-8") as f:
                struct = json.load(f)
            pattern_bits = [feature['bits'] for feature in struct["NET_STLC"]]
            ai_engine.set_structure(pattern_bits, struct["MLP_LAYERS"], struct["ACTIVE_FEATURES"])
            ai_engine.load_weights(cpp_weights_path)
            print(f"[{name}] モデル gen{gen_num} をロードしました。")
        except Exception as e:
            print(f"[ERROR] {name} のモデルロードに失敗しました (gen{gen_num}): {e}")
            sys.exit(1)
    else:
        # gen_num が 0 以下の場合は学習前（石数評価）のエンジン
        pattern_bits = [feature['bits'] for feature in cf.NET_STLC]
        ai_engine.set_structure(pattern_bits, cf.MLP_LAYERS, cf.ACTIVE_FEATURES)
        print(f"[{name}] デフォルトエンジン (石数評価) をロードしました。")
        
    return ai_engine

def generate_random_board(moves_count):
    """初期配置から指定された手数だけランダムに打った盤面を生成する"""
    board = ((1 << 28) | (1 << 35), (1 << 27) | (1 << 36))
    order = 1
    played = 0
    
    while played < moves_count:
        legal = utils.get_valid_moves(board, order)
        if legal != 0:
            moves = list(utils.iter_set_bits(legal))
            put_pos = random.choice(moves)
            board, finish_flag, pass_flag = utils.put_stone(board, put_pos, order)
            
            if finish_flag != 'CONTINUE':
                break
            if not pass_flag:
                order *= -1
            played += 1
        else:
            order *= -1
            
    return board, order

def play_match(engine_black, engine_white, start_board, start_order):
    """指定された盤面・手番から対局を最後まで進め、黒白の最終石数を返す"""
    board = start_board
    order = start_order
    
    while True:
        legal = utils.get_valid_moves(board, order)
        if legal != 0:
            empty_cnt = 64 - (board[0].bit_count() + board[1].bit_count())
            player_bb = board[0] if order == 1 else board[1]
            opponent_bb = board[1] if order == 1 else board[0]
            
            # 手番に応じたエンジンを選択
            current_engine = engine_black if order == 1 else engine_white
            
            # 探索 (終盤は完全読み切り、それ以外は通常探索)
            if empty_cnt <= FULL_DEPTH:
                best_pos, _ = current_engine.best_move(player_bb, opponent_bb, FULL_DEPTH)
            else:
                best_pos, _ = current_engine.best_move(player_bb, opponent_bb, DEPTH)
                
            board, finish_flag, pass_flag = utils.put_stone(board, best_pos, order)
            
            if finish_flag != 'CONTINUE':
                break
            if not pass_flag:
                order *= -1
        else:
            order *= -1
            
    black_stones = board[0].bit_count()
    white_stones = board[1].bit_count()
    return black_stones, white_stones

def evaluate_models():
    engine_A = load_engine(GEN_A, "Model A")
    engine_B = load_engine(GEN_B, "Model B")
    
    a_wins = 0
    b_wins = 0
    draws = 0
    
    print("\n" + "="*50)
    print(f" 対戦開始: Model A (gen{GEN_A}) vs Model B (gen{GEN_B}) (深さ: {DEPTH}, 読み切り深さ: {FULL_DEPTH})")
    print(f" サイクル数: {NUM_CYCLES} (合計 {NUM_CYCLES * 2} 試合)")
    print(f" ランダム進行: 初期 {RANDOM_MOVES} 手")
    print("="*50 + "\n")
    
    start_time = time.time()
    diffs = []
    
    for i in range(NUM_CYCLES):
        # 1. 指定手数だけランダムに打った共通の初期盤面を生成
        start_board, start_order = generate_random_board(RANDOM_MOVES)
        
        # 試合1: Model A(黒) vs Model B(白)
        engine_A.clear_tt()
        engine_B.clear_tt()
        b1, w1 = play_match(engine_A, engine_B, start_board, start_order)
        a_stones_1, b_stones_1 = b1, w1
        diff_1 = a_stones_1 - b_stones_1
        
        if diff_1 > 0: a_wins += 1
        elif diff_1 < 0: b_wins += 1
        else: draws += 1
        
        # 試合2: Model B(黒) vs Model A(白)
        engine_A.clear_tt()
        engine_B.clear_tt()
        b2, w2 = play_match(engine_B, engine_A, start_board, start_order)
        b_stones_2, a_stones_2 = b2, w2
        diff_2 = a_stones_2 - b_stones_2
        
        if diff_2 > 0: a_wins += 1
        elif diff_2 < 0: b_wins += 1
        else: draws += 1
        
        # 1サイクルの集計（1試合基準）
        cycle_diff = (diff_1 + diff_2) / 2
        diffs.append(cycle_diff)
        left_time = int((time.time() - start_time) / (i + 1) * (NUM_CYCLES - (i + 1)))
        left_hour = left_time // 3600
        left_minute = min(59, math.ceil((left_time - left_hour * 3600) / 60))
        print(f"\rCycle {i+1:3d}/{NUM_CYCLES} | Aの勝敗: {a_wins:3d}勝 {b_wins:3d}敗 {draws:2d}分 | 1試合平均石差(A目線): {(sum(diffs) / (i + 1)):+6.2f} | 残り時間: {left_hour} 時間 {left_minute} 分", end="")
        
    print("\n" + "="*50)
    print(" 最終対戦結果")
    print("="*50)
    print(f" Model A (gen{GEN_A})  : {a_wins} 勝")
    print(f" Model B (gen{GEN_B})  : {b_wins} 勝")
    print(f" 引き分け         : {draws} 分")
    print("-" * 50)
    win_rate = ((a_wins + draws / 2) / (NUM_CYCLES * 2)) * 100
    print(f" Model A の勝率      : {win_rate:.1f} %")
    avg = sum(diffs) / NUM_CYCLES
    print(f" 1試合平均石差        : {avg:+.2f} 石 (Model A目線)")
    std = math.sqrt(sum([(d - avg) ** 2 for d in diffs]) / NUM_CYCLES)
    print(f" 1試合石差の標準偏差  : {std:.2f} 石")
    print("="*50)

if __name__ == '__main__':
    evaluate_models()