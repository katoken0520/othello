import os
import random
import pickle
import json
import time
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine # ルートにある engine.pyd を読み込む
from othello_env import config as cf
from othello_env import utils

def play_selfplay_game(ai_engine):
    board = ((1 << 28) | (1 << 35), (1 << 27) | (1 << 36))
    order = 1
    history = []
    early_move = random.randint(cf.EARLY_MOVE[0], cf.EARLY_MOVE[1])
    
    while True:
        legal = utils.get_valid_moves(board, order)
        if legal != 0:
            empty_cnt = 64 - (board[0].bit_count() + board[1].bit_count())
            player_bb = board[0] if order == 1 else board[1]
            opponent_bb = board[1] if order == 1 else board[0]
            
            # --- 【TD学習のコア】実際の着手を決める前に、AIに全力探索させて「真の評価値」を取得する ---
            if empty_cnt <= cf.FULL_DEPTH:
                best_pos, search_score = ai_engine.best_move(player_bb, opponent_bb, cf.FULL_DEPTH)
            else:
                best_pos, search_score = ai_engine.best_move(player_bb, opponent_bb, cf.DATA_DEPTH)

            # 終盤の完全読み切りスコア(±10000)の正規化
            # C++エンジンは勝敗確定時に ±10000 の下駄を履かせるため、純粋な石差に戻す（NNの勾配爆発を防ぐため）
            if search_score > 9000:
                search_score -= 10000
            elif search_score < -9000:
                search_score += 10000

            # ターゲットを黒番から見た評価値に変換 (NNの出力形式に合わせるため)
            history.append({'board': board, 'order': order, 'score': search_score})

            # --- 実際の着手（探索結果を無視して多様な局面を経験させるランダム分岐） ---
            if empty_cnt > 60 - early_move:
                moves = list(utils.iter_set_bits(legal))
                put_pos = random.choice(moves)
            else:
                current_epsilon = max(0, (empty_cnt - cf.FULL_DEPTH) / (60 - early_move - cf.FULL_DEPTH) * cf.EPSILON)
                if random.random() < current_epsilon:
                    moves = list(utils.iter_set_bits(legal))
                    put_pos = random.choice(moves)
                else:
                    put_pos = best_pos # 探索で見つけた最善手を採用
            
            board, finish_flag, pass_flag = utils.put_stone(board, put_pos, order)
            
            if finish_flag != 'CONTINUE':
                break
            if not pass_flag:
                order *= -1
        else:
            order *= -1 

    # 最終的な石差（勝敗確認用）
    stone_diff = board[0].bit_count() - board[1].bit_count()
    
    result_data = []
    for h in history:
        # 探索スコアと最終石差をハイブリッドしてターゲットを作成
        relative_stone_diff = stone_diff * h['order']
        hybrid_target = cf.LAMBDA * h['score'] + (1.0 - cf.LAMBDA) * relative_stone_diff
        result_data.append((h['board'], h['order'], hybrid_target))
        
    return result_data, stone_diff

def generate_selfplay_data():
    all_data = []
    
    # 最新のモデル世代 (M) と 最新のデータ世代 (N) を取得
    latest_model_gen = utils.get_latest_model_gen()
    latest_data_gen = utils.get_latest_data_gen()
    
    print("C++ AIエンジンを初期化中...")
    ai_engine = engine.SearchEngine()
    
    # 最新モデルが存在する場合は、そのフォルダ内から構造を読み込む
    if latest_model_gen > 0:
        model_dir = os.path.join(cf.BASE_DIR, "data", "learned", f"gen{latest_model_gen}")
        structure_path = os.path.join(model_dir, "network_structure.json")
        cpp_weights_path = os.path.join(model_dir, "cpp_weights.bin")
        
        with open(structure_path, "r", encoding="utf-8") as f:
            struct = json.load(f)
            
        pattern_bits = [feature['bits'] for feature in struct["NET_STLC"]]
        ai_engine.set_structure(pattern_bits, struct["MLP_LAYERS"], struct["ACTIVE_FEATURES"])
        ai_engine.load_weights(cpp_weights_path)
        print(f"最新のモデル世代 [gen{latest_model_gen}] の構造と重みを自動復元してロードしました．")
    else:
        # モデルが1つもない最初の初回起動時
        pattern_bits = [feature['bits'] for feature in cf.NET_STLC]
        ai_engine.set_structure(pattern_bits, cf.MLP_LAYERS, cf.ACTIVE_FEATURES)
        print("学習済みモデルが見つからないため、初期状態(石数評価)で動作します．")
        
    # 定石の読み込み
    if os.path.exists(cf.BOOK_PATH):
        with open(cf.BOOK_PATH, "r", encoding="utf-8") as f:
            book = json.load(f)
            for hash_str, move_data in book.items():
                hash_val = int(hash_str)
                # 登録されている全ての手とスコアをエンジンに渡す（選ぶのはC++の役割）
                for move, score in move_data:
                    ai_engine.add_book_move(hash_val, int(move), float(score))

    print(f"自己対戦を {cf.NUM_GAMES} 局行います．")
    wld_counts = {'black': 0, 'white': 0, 'draw': 0}
    start_time = time.time()
    
    for i in range(cf.NUM_GAMES):
        history, diff = play_selfplay_game(ai_engine)
        all_data.extend(history)
        
        if diff > 0: wld_counts['black'] += 1
        elif diff < 0: wld_counts['white'] += 1
        else: wld_counts['draw'] += 1
        
        left_time = int((time.time() - start_time) / (i + 1) * (cf.NUM_GAMES - (i + 1)))
        left_hour = left_time // 3600
        left_minute = min(59, math.ceil((left_time - left_hour * 3600) / 60))
        print(f"\r---- {i+1}/{cf.NUM_GAMES} 局完了 | 黒:{wld_counts['black']} 白:{wld_counts['white']} 分:{wld_counts['draw']} (あと {left_hour}時間{left_minute}分) ----", end="")
            
    print("\n前回の対局結果を読み込み中...")
    if latest_data_gen > 0:
        prev_pickle_path = os.path.join(cf.BASE_DIR, "data", "teaching_data", f"teaching_gen{latest_data_gen}.pickle")
        try:
            with open(prev_pickle_path, mode='rb') as fi:
                old_data = list(pickle.load(fi))
            print(f"最新のデータ(gen{latest_data_gen})から {len(old_data)} 局面をロードしました．マージします．")
            
            all_data = old_data + all_data
            if len(all_data) > cf.MAX_HISTORY:
                print(f"データ数上限超過のため、古い {len(all_data) - cf.MAX_HISTORY} 局面を破棄します。")
                all_data = all_data[-cf.MAX_HISTORY:]
        except Exception as e:
            print(f"過去データのマージに失敗しました: {e}")


    new_pickle_path = os.path.join(cf.BASE_DIR, "data", "teaching_data", f"teaching_gen{latest_data_gen + 1}.pickle")
    os.makedirs(os.path.dirname(new_pickle_path), exist_ok=True)
    with open(new_pickle_path, mode='wb') as fo:
        pickle.dump(all_data, fo)
    print(f"新世代の対局データが作成されました: [teaching_gen{latest_data_gen + 1}.pickle]")

if __name__ == '__main__':
    generate_selfplay_data()