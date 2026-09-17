import engine
from . import config as cf
import torch.nn as nn
import os
import glob
import re

def get_valid_moves(board, order):
    """C++の関数を呼び出して合法手ビットボードを返す"""
    player = board[0] if order == 1 else board[1]
    opponent = board[1] if order == 1 else board[0]
    return engine.get_valid_moves(player, opponent)

def iter_set_bits(n):
    """ビットボード(64bit整数)から、1が立っているマスのインデックス(0~63)を生成する"""
    while n:
        b = n & -n
        yield (b.bit_length() - 1)
        n ^= b

def put_stone(board, put_pos, order):
    """C++の関数を利用して石を置き、新しい盤面とパス・終了判定を返す"""
    player = board[0] if order == 1 else board[1]
    opponent = board[1] if order == 1 else board[0]

    put_bit = 1 << put_pos
    # C++で反転する石を計算
    flipped = engine.get_flipped_bb(put_bit, player, opponent)

    new_player = player ^ (put_bit | flipped)
    new_opponent = opponent ^ flipped

    new_board = (new_player, new_opponent) if order == 1 else (new_opponent, new_player)

    # 終了判定・パス判定のために次のターンの合法手を取得
    next_player_moves = engine.get_valid_moves(new_opponent, new_player)
    
    pass_flag = False
    finish_flag = 'CONTINUE'

    if next_player_moves == 0:
        next_opponent_moves = engine.get_valid_moves(new_player, new_opponent)
        if next_opponent_moves == 0:
            # 双方置けない -> ゲーム終了
            black_cnt = new_board[0].bit_count()
            white_cnt = new_board[1].bit_count()
            if black_cnt > white_cnt: finish_flag = 'BLACK'
            elif white_cnt > black_cnt: finish_flag = 'WHITE'
            else: finish_flag = 'DRAW'
        else:
            # 相手（次の手番）だけ置けない -> パス
            pass_flag = True

    return new_board, finish_flag, pass_flag

def make_additional_features(player_bb, opponent_bb):
    feats = []
    # utils.py内に書いてある場合は config.ACTIVE_FEATURES など適宜合わせてください
    for feat_id in cf.ACTIVE_FEATURES: 
        if feat_id == 0:   
            # 自分の合法手（自分を黒番=1として計算させる）
            feats.append(get_valid_moves((player_bb, opponent_bb), 1).bit_count()) 
        elif feat_id == 1: 
            # 相手の合法手（相手を黒番=1として計算させる）
            feats.append(get_valid_moves((opponent_bb, player_bb), 1).bit_count()) 
        elif feat_id == 2: 
            feats.append(player_bb.bit_count()) # 自分の石数
        elif feat_id == 3: 
            feats.append(opponent_bb.bit_count()) # 相手の石数
    return feats

def make_eval_model_template():
    layers = []
    for i in range(len(cf.MLP_LAYERS) - 1):
        layers.append(nn.Linear(cf.MLP_LAYERS[i], cf.MLP_LAYERS[i+1]))
        if i < len(cf.MLP_LAYERS) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# ---- ★新設: フォルダ内の最新世代を自動検索するヘルパー関数群 ----
def get_latest_data_gen():
    """最新の teaching_genX.pickle の X を返す"""
    files = glob.glob(os.path.join(cf.BASE_DIR, "data", "teaching_data", "teaching_gen*.pickle"))
    if not files: return 0
    gens = [int(re.search(r'teaching_gen(\d+)\.pickle', os.path.basename(f)).group(1)) for f in files if re.search(r'teaching_gen(\d+)\.pickle', os.path.basename(f))]
    return max(gens) if gens else 0

def get_latest_model_gen():
    """最新の learned/genX/ の X を返す"""
    dirs = glob.glob(os.path.join(cf.BASE_DIR, "data", "learned", "gen*"))
    if not dirs: return 0
    gens = []
    for d in dirs:
        match = re.search(r'gen(\d+)', os.path.basename(d))
        # フォルダ内に重みバイナリが存在する場合のみカウント
        if match and os.path.exists(os.path.join(d, "cpp_weights.bin")):
            gens.append(int(match.group(1)))
    return max(gens) if gens else 0