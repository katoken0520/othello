import pickle
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import time
import copy 
import os
import sys
import json

# プロジェクトルートを検索パスに追加
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from othello_env import config as cf
from othello_env import utils

# --- データセット定義 ---
class OthelloDataset(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data
        self.converted_data = []
        total = len(self.data)
        print(f"データを変換中... (全 {total} 件)")
        for idx, (board, order, stone_diff) in enumerate(self.data):
            player_bb = board[0] if order == 1 else board[1]
            opponent_bb = board[1] if order == 1 else board[0]
            relative_board = (player_bb, opponent_bb)
            
            # 抽出関数には相対盤面を渡す
            pattern_indices = make_all_pattern_indices(relative_board)
            add_feats = utils.make_additional_features(player_bb, opponent_bb)
            
            # targetは make_data.py 側で既に相対化されているのでそのまま使う
            self.converted_data.append({'indices': pattern_indices,
                                        'features': np.array(add_feats, dtype=np.float32),
                                        'target': np.array([stone_diff], dtype=np.float32)})

    def __len__(self):
        return len(self.converted_data)
    def __getitem__(self, idx):
        item = self.converted_data[idx]
        return item['indices'], item['features'], item['target']

# --- ヘルパー関数 ---
def make_all_pattern_indices(relative_board):
    indices = {}
    for feature in cf.NET_STLC:
        name = feature['name']
        bit_lists = feature['bits']
        idx_list = []
        for bits in bit_lists:
            idx = 0
            power = 1
            for pos in bits:
                mask = 1 << pos
                if (relative_board[0] & mask):     # 自分の石
                    idx += 2 * power
                elif (relative_board[1] & mask):   # 相手の石
                    idx += 0 * power
                else:                              # 空きマス
                    idx += 1 * power
                power *= 3
            idx_list.append(idx)
        indices[name] = torch.tensor(idx_list, dtype=torch.long)
    return indices

# --- 統合ネットワーク定義 ---
class UnifiedOthelloNet(nn.Module):
    def __init__(self):
        super(UnifiedOthelloNet, self).__init__()
        self.embeddings = nn.ModuleDict()
        for feature in cf.NET_STLC:
            name = feature['name']
            n_bits = len(feature['bits'][0])
            num_embeddings = 3 ** n_bits
            emb = nn.Embedding(num_embeddings, 1)
            nn.init.normal_(emb.weight, std=0.01)
            self.embeddings[name] = emb

        # config.pyで定義した動的生成関数を呼ぶ
        self.eval_model = utils.make_eval_model_template()

    def forward(self, patterns_dict, add_features):
        pre_outputs = []
        for feature in cf.NET_STLC:
            name = feature['name']
            indices = patterns_dict[name]
            val = self.embeddings[name](indices)
            summed_val = torch.sum(val, dim=1)
            pre_outputs.append(summed_val)
        combined = torch.cat(pre_outputs + [add_features], dim=1)
        out = self.eval_model(combined)
        return out

# --- 学習実行 ---
def train_model(raw_data, device='cpu', epoch=10, lr=0.001, batch_size=64):
    print('データセットを作成中...')
    full_dataset = OthelloDataset(raw_data)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = UnifiedOthelloNet().to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    print(f'学習を開始します． (学習局面数: {train_size}, 検証局面数: {val_size})')
    best_loss = float('inf')
    best_model_state = None
    start_time = time.time()  
    
    for ep in range(1, epoch + 1):
        # --- Training ---
        model.train()
        train_loss = 0
        count = 0
        for batch_indices, batch_feats, batch_target in train_loader:
            idx_input = {k: v.to(device) for k, v in batch_indices.items()}
            feat_input = batch_feats.to(device)
            target = batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(idx_input, feat_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_target)
            count += len(batch_target)
        avg_train_loss = train_loss / count
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for batch_indices, batch_feats, batch_target in val_loader:
                idx_input = {k: v.to(device) for k, v in batch_indices.items()}
                feat_input = batch_feats.to(device)
                target = batch_target.to(device)
                
                output = model(idx_input, feat_input)
                loss = criterion(output, target)
                val_loss += loss.item() * len(batch_target)
                count += len(batch_target)
        avg_val_loss = val_loss / count

        # Best Model Update
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # エポック終了時に学習率を減衰させる
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        left_time = int((time.time() - start_time) / ep * (epoch - ep))
        left_hour = left_time // 3600
        left_minute = min(59, math.ceil((left_time - left_hour * 3600) / 60))
        print(f"\r---- 進捗: {ep}/{epoch} | Train: {avg_train_loss:.3f} | Val: {avg_val_loss:.3f} (Best: {best_loss:.3f}) | LR: {current_lr:.5f} | (あと {left_hour}時間{left_minute}分) ----", end="")
    
    print("\n学習終了．")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"検証ロスが最小({best_loss:.3f})だった時点のモデルを採用します．")
        
    return model

# --- 保存処理 ---
def save_results(model, gen_num):
    model.cpu()
    # 保存先フォルダ: data/learned/genN/
    gen_dir = os.path.join(cf.BASE_DIR, "data", "learned", f"gen{gen_num}")
    os.makedirs(gen_dir, exist_ok=True)
    
    model_path = os.path.join(gen_dir, "model.pth")
    cpp_weights_path = os.path.join(gen_dir, "cpp_weights.bin")
    structure_path = os.path.join(gen_dir, "network_structure.json")
    
    print(f"評価モデル群をフォルダに保存中: {gen_dir}")
    torch.save(model.eval_model, model_path)
    
    cpp_weights_list = []

    with torch.no_grad():
        # ① 特徴量（パターン）の重みを抽出
        for feature in cf.NET_STLC:
            name = feature['name']
            table = model.embeddings[name].weight.data.numpy().flatten()
            cpp_weights_list.append(table)
            
        # ② 後続のMLP層の重みとバイアスを抽出
        layer_idx = 0
        for _ in range(len(cf.MLP_LAYERS) - 1):
            w = model.eval_model[layer_idx].weight.data.numpy().flatten()
            b = model.eval_model[layer_idx].bias.data.numpy().flatten()
            cpp_weights_list.append(w)
            cpp_weights_list.append(b)
            layer_idx += 2

    # 全て結合してC++用の1つの高速バイナリとして保存
    all_weights = np.concatenate(cpp_weights_list).astype(np.float32)
    all_weights.tofile(cpp_weights_path)
    
    # ネットワーク構造情報をJSON形式で保存
    structure_data = {
        "NET_STLC": cf.NET_STLC,
        "MLP_LAYERS": cf.MLP_LAYERS,
        "ACTIVE_FEATURES": cf.ACTIVE_FEATURES
    }
    with open(structure_path, "w", encoding="utf-8") as f:
        json.dump(structure_data, f, indent=4, ensure_ascii=False)
        
    print("保存がすべて完了しました．")

if __name__ == '__main__':
    # 最新のデータ世代 (N) を探す
    latest_data_gen = utils.get_latest_data_gen()
    latest_model_gen = utils.get_latest_model_gen()
    
    if latest_data_gen > 0:
        pickle_path = os.path.join(cf.BASE_DIR, "data", "teaching_data", f"teaching_gen{latest_data_gen}.pickle")
        print(f"最新の対局データ [teaching_gen{latest_data_gen}.pickle] をロードして学習します")
        
        with open(pickle_path, mode='rb') as fi:
            teaching_data = pickle.load(fi)
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trained_model = train_model(teaching_data, device=device, epoch=cf.EPOCH, batch_size=256)
        
        save_results(trained_model, latest_model_gen + 1)
    else:
        print("学習対象となる対局データ(pickle)が見つかりません．まず make_data.py を実行してください．")