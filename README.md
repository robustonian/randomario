# RandoMario

[![Python version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概要

RandoMario は、OpenAI Gym のスーパーマリオブラザーズ環境 (`gym-super-mario-bros`) を使用して、マリオを自動でプレイさせるプロジェクトです。

2種類のエージェントを搭載しています。

1. **Planner(モデルベース攻略AI)** — 直近フレームの画像からマリオ・敵・穴の位置/速度/加速度を推定し、次の数〜数十ステップの入力プランを複数生成、エミュレータのセーブステートを使った先読みロールアウトで評価して最良プランを実行します。死亡時には原因(穴/敵/接触/時間切れ)を分析してステージ別ハザードメモリ(SQLite)に記録し、同じ失敗を繰り返しません。**ステージ固有のハードコードは一切なく、初見のステージでも共通ロジックで攻略できます。**
2. **Go-Explore(ランダム攻略ベースライン)** — セル分割アーカイブによる「戻ってから探索」方式のランダムエージェント(従来の `test.py`)。

両者の学習データは完全に分離されており(Planner: `db/planner.sqlite` / Random: `pkl/*.pkl`)、比較ダッシュボードで攻略状況を並べて確認できます。

## 特徴

- **モデル予測型プランニング**: 候補プラン(ダッシュ/ジャンプ長・タイミング掃引/待機/後退/水泳ストローク)を毎リプラン時に生成し、セーブステートからの先読みロールアウトで採点
- **画像ベースの状態推定**: スクロール補償フレーム差分による移動体検出・追跡で、敵の位置・速度・加速度と穴(ピット)を画像から推定し、ジャンプ/踏みつけタイミングの候補生成に利用
- **死亡原因分析と再発防止**: 死亡時に原因を分類してハザードDBへ記録。次回以降は該当地点付近で先読み距離と候補数を自動拡大し、失敗プランはブラックリスト化
- **段階的エスカレーション**: 通常時は少数候補で高速、全滅時は精密なタイミング掃引・長い待機・後退を含む広い探索へ自動切替
- **実行中可変の速度制御**: 1x/2x/4x/8x/MAX/SMOOTH をキー操作で切替(エミュレーション精度は不変、描画とウェイトのみ変化)
- **SMOOTHモード(プランナーのデフォルト)**: エミュレーションと思考は裏で最速実行し、録画したフレームをバッファから一定レートで再生。プランナーの「考え込み停止」が画面上から消える(数秒の表示遅延とトレードオフ)
- **刷新されたUI**: ゲーム画面 + ファミコンコントローラー入力可視化 + ビジョン検出オーバーレイ(敵/マリオ/穴)+ プランナー状態パネル
- **比較ダッシュボード**: 全32ステージの Random vs Planner の攻略状況(初クリアエピソード・進捗)をリアルタイム監視
- **並列一括検証**: `verify.py` で全ステージをマルチプロセス・ヘッドレス検証

## 動作デモ

左側にゲーム画面、右側にコントローラー入力と実行状態が表示されます。

![実行画面](./fig/1-1.gif)

## セットアップ

```bash
git clone https://github.com/robustonian/randomario.git
cd randomario
uv sync
```

## 使い方

### モデルベース攻略AI(Planner)

```bash
uv run play.py --agent planner --stage 1-1          # UIあり(初期速度1x)
uv run play.py --agent planner --stage 8-1 --speed max
uv run play.py --agent planner --stage 2-2 --headless   # UIなし最速
```

### リプレイ(記録済みランの再生)

プランナー実行時に各エピソードの入力列が自動記録されます。リプレイモードは、直近でクリアに到達したランを **EP1〜クリアエピソードまでプランニングなしの一定fps** で再生し、最後にクリア祝福を表示します(実行中の速度変更も可能)。

```bash
uv run play.py --agent replay --stage 1-1              # 直近のクリアランを再生
uv run play.py --agent replay --stage 1-1 --speed 4x   # 4倍速で再生
uv run play.py --agent replay --stage 1-1 --run-id 20260708-091616  # ラン指定
```

Rキーで次のエピソードへスキップできます。※入力記録の追加以前に実行されたランは再生できません(プランナーを一度走らせて記録を作ってください)。

### ランダム攻略(Go-Explore)

```bash
uv run play.py --agent explore --stage 1-1 --actions right_only
uv run test.py --stage 1-1                          # 旧CLI互換
```

### 全ステージ検証(ループステージ 4-4, 7-4, 8-4 を除く29ステージ)

```bash
uv run verify.py                     # 全対象ステージを並列検証
uv run verify.py --stages 8-1 7-4 --episodes 60
```

### 比較ダッシュボード

```bash
python3 progress_viewer.py           # 標準ライブラリのみで動作
```

## 実行中の操作

| キー | 動作 |
|------|------|
| 1〜6 | 速度切替 (1x / 2x / 4x / 8x / MAX / SMOOTH) |
| ← → (↑↓) / + - | 速度を1段階変更 |
| P | 一時停止 |
| V | ビジョンオーバーレイ表示切替 |
| R | エピソードをリセット |
| ESC | 終了 |

## アーキテクチャ

```
randomario/
├── mario/
│   ├── env_utils.py      # env直接生成・セーブステート・高速ロールアウト
│   ├── actions.py        # アクションセット定義
│   ├── vision.py         # 画像からの状態推定(位置・速度・加速度・穴)
│   ├── planner.py        # 候補生成 + ロールアウト評価(MPC)
│   ├── memory.py         # 死亡分析メモリ + 結果DB(SQLite)
│   ├── agent_planner.py  # モデルベースエージェントのエピソードループ
│   ├── agent_explore.py  # Go-Explore ランダムエージェント
│   └── ui.py             # 共通Pygame UI(可変速度・オーバーレイ)
├── play.py               # 統合エントリポイント
├── verify.py             # 並列一括検証
├── progress_viewer.py    # Random vs Planner 比較ダッシュボード
├── pkl/                  # ランダム攻略のアーカイブ(Go-Explore)
└── db/                   # プランナーの学習・結果DB(自動生成, gitignore)
```

## ライセンス

MIT
