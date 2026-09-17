# 1. 今後やりたいこと
・TD学習の導入\
・MoveOrderingの強化（キラーヒューリスティック？）\
・`opening_book.json`の追加\
・`othello_core.cpp`の`MAX_TT_MOVES`をなるべく小さくしたい

# 2. モデルの変更手順について
## 2-1. エンジン（C++で作成）のコンパイル
`python .\src\setup.py build_ext --inplace`を実行\
`engine.cp313-win_amd64.pyd`が作成される。ルートディレクトリ（`othello`）にあることを確認。\
以後、Python側では`import engine`として使える

## 2-2. モデルの変更をしたい場合
`othello_env/config.py`のみを変更する\
その後2-1.を実行

## 2-3. 追加特徴量の変更をしたい場合
`src/cpp/othello_core.cpp`のswitch文に`case 5: ...`を足す。\
`othello_env/config.py`の関数に`elif feat_id == 5: ... `を足す。`ACTIVE_FEATURES = [0, 1, 2, 3, 4, 5]`に変更する。\
その後2-1.を実行

# 3. 学習の進め方
## 3-1. 自己対局を行う
`python .\scripts\make_data.py`を実行。自己対局が行われる。\
設定は`othello_env/config.py`で行う。\
学習を進めている感じ、一つ前の世代に勝つためだけの局所最適化が発生している可能性がある。つまり、ここで行う自己対局数はconfig内の`MAX_HISTORY`に対してある程度小さい数にとどめておく必要がある。

## 3-2. モデルの学習を行う
`python .\scripts\learning.py`を実行。モデルが学習される。\
設定は`othello_env/config.py`で行う。

## 3-3. 特徴パターンモデルの重要度の特定
`python .\scripts\check.py`を実行。各特徴パターン（`othello_env/config.py`の`NET_STLC`の部分）の重要度を見ることができる。
モデルにはL2正則化を使っており重要でない特徴パターンについては重みが0に近づくはずだからである。

# 4. オセロ対局
## 4-1. 起動
`python .\scripts\learning.py`を実行。\
設定は`othello_env/config.py`で行う。

# 5. 定石の登録
## 4-1. エディタ
`python .\scripts\book_editor.py`を実行。
一回登録したら初手4パターンのそれぞれについて回転したパターンも自動で登録される。

## 4-2. 今登録されている内容
e6 f4 c3 c4 d3 d6 e3 c2 b3 d2 c5 (12手目でb5を選んだパターンは以降すべて登録済み。f5のパターンはまだ)