import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from othello_env import config as cf
from othello_env import utils

# ==========================================
# テスト設定
# ==========================================
GEN = None  # パラメータをチェックするモデルの数字（整数）。Noneなら最新のモデルをチェックする。
# ==========================================

def analyze_feature_importances():
    # 最新のモデル世代を自動取得
    if GEN is None:
        latest_model_gen = utils.get_latest_model_gen()
        if latest_model_gen == 0:
            print("学習済みモデルが見つかりません。学習を1回以上実行してください。")
            return

    model_dir = os.path.join(cf.BASE_DIR, "data", "learned", f"gen{latest_model_gen}")
    structure_path = os.path.join(model_dir, "network_structure.json")
    cpp_weights_path = os.path.join(model_dir, "cpp_weights.bin")

    # 構造データと重みバイナリをロード
    with open(structure_path, "r", encoding="utf-8") as f:
        struct = json.load(f)
    weights = np.fromfile(cpp_weights_path, dtype=np.float32)

    print(f"--- 特徴量の重要度ランキング (AI世代: gen{latest_model_gen}) ---")
    importances = []

    # バイナリデータからパターンごとの重みブロックを切り出して分散を計算
    offset = 0
    for feature in struct["NET_STLC"]:
        name = feature['name']
        n_bits = len(feature['bits'][0])
        size = 3 ** n_bits  # 例: 8マスなら 3^8 = 6561個のパラメータ
        
        feature_weights = weights[offset : offset + size]
        weight_variance = np.var(feature_weights)
        importances.append((name, weight_variance))
        
        offset += size

    # 分散が大きい順にソートして表示
    importances.sort(key=lambda x: x[1], reverse=True)
    for i, (name, var) in enumerate(importances):
        print(f"{i+1}位: {name} (重要度スコア: {var:.5f})")

if __name__ == '__main__':
    analyze_feature_importances()