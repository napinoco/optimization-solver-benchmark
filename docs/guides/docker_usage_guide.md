# Docker使用方法ガイド - Optimization Solver Benchmark System

Docker環境での最適化ソルバーベンチマークシステムの使用方法を説明します。

## 🚀 クイックスタート

### 1. 前提条件

必要なソフトウェア：
- **Docker Desktop** (macOS/Windows) または **Docker Engine** (Linux)
- **Git** (サブモジュール対応)
- **MATLAB Individual License** (MATLAB ソルバー使用時)

### 2. リポジトリの取得

```bash
# サブモジュールと一緒にクローン
git clone --recursive https://github.com/your-username/optimization-solver-benchmark.git
cd optimization-solver-benchmark
```

### 3. MATLAB ライセンス設定（オプション）

```bash
# ライセンス設定の確認
./scripts/setup-matlab-license.sh --check

# 初回セットアップ
./scripts/setup-matlab-license.sh --setup
```

### 4. Docker イメージのビルド

```bash
# Docker イメージをビルド
./scripts/docker-run.sh --build
```

### 5. ベンチマーク実行

```bash
# システム検証
./scripts/docker-run.sh -- python main.py --validate

# 全ベンチマーク実行
./scripts/docker-run.sh -- python main.py --all

# 特定の問題セット実行
./scripts/docker-run.sh -- python main.py --benchmark --library_names DIMACS
```

---

## 📋 詳細な使用方法

### Docker 実行スクリプト

`./scripts/docker-run.sh` は Docker コンテナでの実行を簡単にするヘルパースクリプトです。

#### 基本的な使用方法

```bash
# 構文
./scripts/docker-run.sh [オプション] [-- コマンド]
```

#### 主要オプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--memory SIZE` | メモリ制限 | 16g |
| `--cpu CORES` | CPU制限 | 4.0 |
| `--timeout SECONDS` | ベンチマークタイムアウト | 120 |
| `--build` | 実行前にイメージをビルド | - |
| `--clean` | 既存コンテナを削除してから実行 | - |
| `--shell` | 対話的シェルを起動 | - |
| `--matlab` | MATLAB モードを強制 | auto-detect |
| `--verbose` | 詳細ログ出力 | - |
| `--dry-run` | コマンドを表示するだけ | - |

#### 実行例

```bash
# 対話的シェル
./scripts/docker-run.sh --shell

# システム検証
./scripts/docker-run.sh -- python main.py --validate

# 全ベンチマーク実行（16GBメモリ）
./scripts/docker-run.sh --memory 16g -- python main.py --all

# DIMACS問題のみ実行（10分タイムアウト）
./scripts/docker-run.sh --timeout 600 -- python main.py --benchmark --library_names DIMACS

# 特定問題とソルバー
./scripts/docker-run.sh -- python main.py --benchmark --problems nb --solvers cvxpy_clarabel

# レポート生成のみ
./scripts/docker-run.sh -- python main.py --report
```

---

## 🔧 MATLAB ライセンス設定

### ライセンス要件

- **MATLAB Individual License** (Home License)
- ホストマシンでのMATLABアクティベーション完了
- `~/.matlab/` ディレクトリへのライセンスファイル配置

### セットアップ手順

#### 1. ライセンス状況確認

```bash
./scripts/setup-matlab-license.sh --check
```

出力例:
```
✓ Host MATLAB: Available
✓ License Directory: /Users/username/.matlab
✓ License Versions: 1 found
✓ License Files: 1 found
✓ Docker Ready: Yes - license can be used in containers
```

#### 2. 初回セットアップ

```bash
./scripts/setup-matlab-license.sh --setup
```

このコマンドは以下を実行します：
- MATLAB インストール確認
- ライセンスファイル検証
- Docker マウント設定確認
- 設定ガイダンス表示

#### 3. トラブルシューティング

```bash
# ライセンス検証
./scripts/setup-matlab-license.sh --validate

# 一時ファイルクリーンアップ
./scripts/setup-matlab-license.sh --clean
```

### ライセンスファイル構造

```
~/.matlab/
├── R2024a_licenses/
│   └── license.lic          # ライセンスファイル（必須）
├── MathWorks/
│   └── MATLAB/
│       └── R2024a/
│           └── licenses/    # アクティベーションデータ
└── .matlab_license_token    # ライセンストークン
```

---

## 🐳 Docker Compose 使用方法

### 基本的な使用

```bash
# サービス起動
cd docker
docker-compose up benchmark

# バックグラウンド実行
docker-compose up -d benchmark

# ログ確認
docker-compose logs -f benchmark

# 停止
docker-compose down
```

### 環境変数カスタマイズ

```bash
# .envファイルを作成
cp docker/.env.example docker/.env

# 必要に応じて編集
vim docker/.env
```

`.env` ファイル例:
```env
# リソース制限
MEMORY_LIMIT=16g
MEMORY_RESERVATION=4g
CPU_LIMIT=4.0
CPU_RESERVATION=1.0

# ベンチマーク設定
BENCHMARK_TIMEOUT=120
CONTAINER_ENV=development

# その他の設定
TZ=Asia/Tokyo
```

### 開発用シェル

```bash
# 開発用プロファイルで対話的シェル起動
docker-compose --profile dev up shell
```

---

## 📊 メモリとリソース管理

### メモリ制限の重要性と動作

Docker環境では確実なメモリ制限により、以下を実現します：

#### ✅ 実装済み改善点
- **正確なメモリ検出**: コンテナ内でもcgroup制限値を正確に報告（ホスト16GB環境で7.8GB表示の問題を解決）
- **制御されたメモリ管理**: `--memory-swap="16g"`でスワップ使用量を制限
- **ホスト保護**: コンテナ内でのOOM Killer動作により、ホストシステムへの影響を防止

#### 期待される動作
- **正常ケース**: ソルバーがメモリ制限内で完了
- **メモリ不足ケース**: 
  - コンテナのメモリ制限に達する
  - コンテナ内のOOM Killerが動作してプロセスを終了
  - **ホストシステムは影響を受けない**
  - 結果として`SolverResult.status = 'SIGKILL'`が適切に記録される

**重要**: SIGKILLは「防がれる」のではなく、「制御下で発生」します。これによりホスト環境を保護しながら問題のあるソルバーを分析できます。

### 推奨設定

| 用途 | メモリ制限 | CPU制限 | 説明 |
|------|-----------|---------|------|
| 軽量テスト | 4g | 2.0 | 小規模問題のテスト |
| 標準ベンチマーク | 16g | 4.0 | ほとんどの問題に対応 |
| 大規模問題 | 32g | 8.0 | 極大規模SDP問題など |

### メモリ不足対応

```bash
# メモリ制限を増加
./scripts/docker-run.sh --memory 32g -- python main.py --all

# より長いタイムアウト設定
./scripts/docker-run.sh --timeout 1800 -- python main.py --benchmark --problems difficult_problem

# 問題を分割して実行
./scripts/docker-run.sh -- python main.py --benchmark --library_names DIMACS
./scripts/docker-run.sh -- python main.py --benchmark --library_names SDPLIB
```

---

## 🔍 トラブルシューティング

### Docker 関連の問題

#### Docker デーモンが起動していない
```bash
# macOS/Windows
Docker Desktop を起動

# Linux
sudo systemctl start docker
```

#### ビルドエラー
```bash
# キャッシュをクリアしてリビルド
docker system prune -f
./scripts/docker-run.sh --build

# SCIP を除外してより高速にビルド
./scripts/docker-run.sh --build --no-scip
```

#### ✅ 解決済みビルド問題
- **Timezone プロンプト**: `ENV DEBIAN_FRONTEND=noninteractive`により対話型設定をスキップ
- **Group ID 競合**: macOSの標準GID 20との競合を自動解決
- **MATLAB バージョン**: R2024a への統一により一貫性を確保

#### 権限エラー
```bash
# ユーザーをdockerグループに追加（Linux）
sudo usermod -aG docker $USER
# ログアウト・ログインが必要
```

### MATLAB ライセンス問題

#### ライセンスファイルが見つからない
```bash
# MATLAB を一度起動してライセンスを有効化
matlab

# ライセンス状況を再確認
./scripts/setup-matlab-license.sh --check
```

#### ライセンス認証エラー
```bash
# MATLABライセンスマネージャーで再認証
matlab
# Help > Licensing > Activate Software

# 一時ファイルをクリーンアップ
./scripts/setup-matlab-license.sh --clean
```

### パフォーマンス問題

#### 実行が遅い
```bash
# CPU制限を増加
./scripts/docker-run.sh --cpu 8.0 -- python main.py --all

# 並列度を調整（main.pyのオプション確認）
./scripts/docker-run.sh -- python main.py --help
```

#### メモリ不足でクラッシュ
```bash
# メモリ制限を増加
./scripts/docker-run.sh --memory 32g -- python main.py --all

# 問題を分割実行
./scripts/docker-run.sh -- python main.py --benchmark --problems small_problem
```

---

## 🚀 高度な使用方法

### CI/CD 環境での使用

GitHub Actions では自動的にDockerベースの検証が実行されます：

```yaml
# .github/workflows/validate.yml
# Native Python 検証 + Docker コンテナ検証の両方実行
```

### カスタムイメージの作成

```bash
# ベースイメージからカスタマイズ
docker build -f docker/Dockerfile -t my-solver-benchmark .

# カスタムイメージでの実行
DOCKER_IMAGE=my-solver-benchmark ./scripts/docker-run.sh -- python main.py --validate
```

### 本番環境デプロイ

```bash
# 本番用設定
cp docker/.env.example docker/.env.prod
# CONTAINER_ENV=production に設定

# 本番環境での実行
docker-compose --env-file docker/.env.prod up benchmark
```

---

## 📝 FAQ

### Q: Octaveは必要ですか？
A: いいえ。このシステムはMATLAB Individual Licenseの使用を前提としており、Octaveはサポートしていません。

### Q: Macでメモリ制限が効かない場合は？
A: Dockerコンテナのメモリ制限により確実に制限されます。ulimitの代替として機能します。

### Q: どの程度のメモリが必要ですか？
A: 一般的な問題では16GBで十分です。大規模SDP問題（>50K変数）では32GB以上を推奨します。

### Q: MATLAB ライセンスをCI環境で使用できますか？
A: Individual Licenseの利用規約に従ってください。通常、個人使用に限定されます。

### Q: 結果の再現性は保証されますか？
A: Docker環境により高い再現性を実現していますが、完全な決定論的実行ではありません。

### Q: コンテナ内でメモリが正しく表示されない場合は？
A: ✅ 解決済み - environment_info.pyでcgroup制限値を使用するよう修正済みです。16GB制限のコンテナで正確に16GBと表示されます。

### Q: SIGKILLエラーは問題ですか？
A: SIGKILLはエラーではなく、メモリ制限による正常な終了です。Docker化により、ホストへの影響なくソルバーの問題を特定できます。

---

## 📚 関連ドキュメント

- [基本設計書](../development/basic_design.md) - システム概要と設計思想
- [詳細設計書](../development/detail_design.md) - 技術仕様と実装詳細
- [Docker設計書](../development/docker_design.md) - Docker化の設計方針
- [開発規約](../development/conventions.md) - 開発ガイドライン

---

*最終更新: 2025年8月*