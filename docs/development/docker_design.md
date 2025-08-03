# Docker化設計書 - Optimization Solver Benchmark System

## 1. 背景と目的

### 1.1 現状の課題
- **メモリ管理の問題**: 品質の悪いソルバーがホストのメモリとスワップを使い尽くし、SIGKILLになるケースが散見される
- **Mac環境での制限**: macOSではulimitによるメモリ制限が機能しない
- **環境汚染リスク**: セグメンテーションフォルトなど、ソルバーのクラッシュがホスト環境に影響を与える可能性
- **環境の再現性**: 異なる開発環境での実行結果の一貫性確保が困難

### 1.2 Docker化の目的
- **リソース隔離**: コンテナレベルでのメモリ・CPU制限による確実なリソース管理
- **環境保護**: ソルバーの異常動作からホスト環境を保護
- **再現性向上**: 統一された実行環境による結果の一貫性確保
- **配布容易性**: 他の研究者との環境共有の簡素化

---

## 2. 要件定義

### 2.1 機能要件
1. **メモリ制限**
   - コンテナレベルでのメモリ制限設定
   - メモリ不足時の適切なエラーハンドリング
   - 既存のSIGKILL検出ロジックとの統合

2. **MATLABサポート**
   - MATLAB Individualライセンス（Homeライセンス）のアクティベーション
   - ライセンスファイルの永続化

3. **実行インターフェース**
   - main.pyの全ての実行時引数サポート
   - ホスト環境からの透過的な実行
   - 結果ファイルのホストへの適切な同期

4. **CI/CD統合**
   - GitHub Actionsでのコンテナ実行
   - ライセンス情報の安全な管理
   - 既存のワークフローとの互換性

### 2.2 非機能要件
1. **パフォーマンス**
   - コンテナ化によるオーバーヘッドの最小化
   - 起動時間の最適化

2. **保守性**
   - 設定の外部化による柔軟性確保
   - ログとデバッグ情報の適切な出力

3. **セキュリティ**
   - ライセンス情報の安全な管理
   - 最小権限原則の適用

---

## 3. アーキテクチャ設計

### 3.1 コンテナ化戦略

#### 選択: 親プロセスレベルコンテナ化

```
Host System
│
├─ docker run
│
└─ Container
    │
    ├─ main.py (親プロセス)
    │
    ├─ Python Subprocess
    │   ├─ python_solver_runner.py
    │   └─ 各種Pythonソルバー
    │
    └─ MATLAB Subprocess
        ├─ matlab_solver_runner.m
        └─ SeDuMi/SDPT3
```

**選択理由:**
- サブプロセスごとのコンテナ化と比較して、オーバーヘッドが大幅に削減
- MATLABライセンス管理が単一コンテナ内で完結
- 既存のサブプロセスアーキテクチャをそのまま活用可能
- 環境変数やファイルシステムの共有が容易

### 3.2 ディレクトリ構造

```
optimization-solver-benchmark/
├── docker/
│   ├── Dockerfile          # マルチステージビルド定義
│   ├── docker-compose.yml  # 開発環境用設定
│   └── entrypoint.sh      # コンテナ起動スクリプト
├── scripts/
│   ├── docker-run.sh      # コンテナ実行ヘルパー
│   └── setup-matlab-license.sh  # ライセンス設定補助
└── .dockerignore          # ビルド除外設定
```

### 3.3 ボリュームマウント設計

```yaml
volumes:
  # MATLABライセンス（読み取り専用）
  - ~/.matlab:/home/solver/.matlab:ro
  
  # 問題ファイル（読み取り専用）
  - ./problems:/app/problems:ro
  
  # データベースと結果（読み書き可能）
  - ./database:/app/database:rw
  - ./docs:/app/docs:rw
  
  # 設定ファイル（読み取り専用）
  - ./config:/app/config:ro
```

---

## 4. 実装設計

### 4.1 Dockerfile設計

```dockerfile
# マルチステージビルド採用
# Stage 1: Python依存関係のビルド
FROM python:3.12-slim as python-deps

# Stage 2: MATLAB Runtime環境
FROM mathworks/matlab:r2024a as matlab-env

# Stage 3: 最終実行環境
FROM ubuntu:22.04
# Python + MATLAB Runtime統合環境
```

### 4.2 メモリ制限実装

#### Dockerレベルの制限
```bash
docker run --memory="8g" --memory-swap="8g" ...
```

#### 検出と報告の実装済み修正
- **メモリ制限検出の修正**: environment_info.pyでpsutilではなくcgroup制限値を使用
  - コンテナ内では`/sys/fs/cgroup/memory/memory.limit_in_bytes` (cgroup v1)
  - または`/sys/fs/cgroup/memory.max` (cgroup v2) を読み取り
  - ホストメモリではなくコンテナ制限値を正確に報告
- **メモリスワップ制限**: `--memory-swap="$MEMORY"`でスワップを制限値と同一に設定
- **OOM Killer設定**: `--oom-kill-disable=false`で制御されたコンテナ内終了を実現
  - ホストのSIGKILL（制御不能）→コンテナのSIGKILL（制御可能）への変更
  - ホストシステム保護を目的とした設計

### 4.3 MATLABライセンス管理

#### ライセンスファイル構造
```
~/.matlab/
├── R2024a_licenses/
│   └── license.lic
├── MathWorks/
│   └── MATLAB/
│       └── R2024a/
│           └── licenses/
└── .matlab_license_token
```

#### アクティベーション戦略
1. 初回実行時：ホストでアクティベーション実施
2. コンテナ実行時：ライセンスディレクトリをマウント
3. CI環境：Base64エンコードしたライセンスをSecretから復元

#### 実装済みMATLABバージョン統一
- **バージョン更新**: R2023b → R2024a への全面移行
  - Dockerfile: `FROM mathworks/matlab:r2024a`
  - ライセンスパス: `/home/solver/.matlab/R2024a_licenses/license.lic`
  - 環境変数: `MLM_LICENSE_FILE=/home/solver/.matlab/R2024a_licenses/license.lic`

### 4.4 追加の実装修正

#### Dockerビルド安定化修正
- **Timezone問題の解決**: `ENV DEBIAN_FRONTEND=noninteractive`をDockerfileに追加
  - Ubuntu 22.04での対話型timezone設定をスキップ
  - ビルド時の hang 問題を根本解決
- **Group ID競合問題の解決**: 既存のGIDを適切に処理
  - macOSの標準GID 20 (staff group)との競合を回避
  - `getent group ${GROUP_ID} || groupadd`パターンで安全な作成

#### Docker実行時の制御改善
- **メモリスワップ制御**: `--memory-swap="$MEMORY"`でスワップ使用量を制限
- **OOM Killer制御**: `--oom-kill-disable=false`で適切な終了動作
- **セキュリティ強化**: `--security-opt no-new-privileges:true`

### 4.5 environment_info.py拡張

```python
def get_memory_info() -> Dict[str, Any]:
    """メモリ情報取得（コンテナ対応済み実装）"""
    memory = psutil.virtual_memory()
    result = {
        "total": memory.total,
        "available": memory.available,
        "total_gb": round(memory.total / (1024**3), 2),
        # ... 他のメモリ情報
    }
    
    # コンテナ内ではcgroupメモリ制限を使用（実装済み修正）
    if os.path.exists('/.dockerenv'):
        container_info = get_container_info()
        if container_info.get('memory_limit'):
            cgroup_total = container_info['memory_limit']
            # cgroupの制限値を権威ある値として使用
            if cgroup_total != memory.total:
                result["total"] = cgroup_total
                result["total_gb"] = round(cgroup_total / (1024**3), 2)
                # 比例配分で他の値を再計算
                proportion_available = memory.available / memory.total
                result["available"] = int(cgroup_total * proportion_available)
                result["available_gb"] = round(result["available"] / (1024**3), 2)
    
    return result

def get_container_info() -> Dict[str, Any]:
    """コンテナ環境情報の収集（完全実装版）"""
    container_info = {
        'is_container': False,
        'container_type': None,
        'memory_limit': None,
        'memory_limit_gb': None,
        'cpu_limit': None
    }
    
    # Docker検出
    if os.path.exists('/.dockerenv'):
        container_info['is_container'] = True
        container_info['container_type'] = 'docker'
        
        # メモリ制限取得（cgroup v1/v2対応）
        memory_limit_files = [
            '/sys/fs/cgroup/memory/memory.limit_in_bytes',  # cgroup v1
            '/sys/fs/cgroup/memory.max'  # cgroup v2
        ]
        
        for limit_file in memory_limit_files:
            if os.path.exists(limit_file):
                with open(limit_file, 'r') as f:
                    limit = int(f.read().strip())
                    if limit < 9223372036854775807:  # 実際の制限値
                        container_info['memory_limit'] = limit
                        container_info['memory_limit_gb'] = round(limit / (1024**3), 2)
                        break
    
    return container_info
```

---

## 5. 実行方法設計

### 5.1 開発環境での実行

```bash
# 初回セットアップ
./scripts/setup-matlab-license.sh

# ベンチマーク実行（メモリ8GB制限）
./scripts/docker-run.sh --memory 8g -- python main.py --all

# 特定の問題セットを実行
./scripts/docker-run.sh -- python main.py --benchmark --library_names DIMACS

# タイムアウト設定付き実行
./scripts/docker-run.sh -- python main.py --all --timeout 600
```

### 5.2 CI環境での実行

```yaml
# GitHub Actions workflow
steps:
  - name: Setup MATLAB License
    run: |
      echo "${{ secrets.MATLAB_LICENSE_B64 }}" | base64 -d > license.lic
      mkdir -p ~/.matlab/R2024a_licenses
      mv license.lic ~/.matlab/R2024a_licenses/
      
  - name: Run Benchmarks in Container
    run: |
      docker-compose run --rm benchmark python main.py --validate
```

---

## 6. 考慮事項とリスク

### 6.1 技術的考慮事項

1. **Dockerイメージサイズ**
   - MATLAB Runtimeは数GB規模
   - マルチステージビルドによる最適化が必要
   - イメージレジストリの選択（Docker Hub vs GitHub Container Registry）

2. **プラットフォーム互換性**
   - ARM64（Apple Silicon）とx86_64の両対応
   - プラットフォーム別のイメージビルド

3. **ファイルパーミッション**
   - コンテナ内ユーザーとホストユーザーのUID/GID調整
   - 結果ファイルの適切な権限設定

### 6.2 運用上のリスクと対策

1. **ライセンス管理**
   - リスク：ライセンスファイルの誤った共有
   - 対策：.gitignoreでの確実な除外、ドキュメントでの注意喚起

2. **パフォーマンス劣化**
   - リスク：コンテナ化によるオーバーヘッド
   - 対策：ベースラインベンチマークによる性能比較

3. **デバッグの困難性**
   - リスク：コンテナ内での問題調査が困難
   - 対策：適切なログ出力、デバッグモードの実装

---

## 7. 移行計画

### 7.1 段階的移行

1. **Phase 1**: Docker環境の構築とテスト ✅
   - Dockerfile作成
   - 基本的な動作確認
   - 開発環境でのテスト

2. **Phase 2**: 機能統合と修正実装 ✅
   - environment_info.py拡張（cgroup対応メモリ検出）
   - メモリ制限の検証と修正
   - MATLABライセンス統合（R2024a対応）
   - Dockerビルド安定化（timezone問題解決）

3. **Phase 3**: CI/CD統合
   - GitHub Actions更新
   - ドキュメント整備
   - 本番環境への適用

### 7.2 互換性維持

- 既存のネイティブ実行も引き続きサポート
- 実行モード（native/container）の自動検出
- 段階的な移行を可能にする設計

---

## 8. 実装完了後の重要な注意事項

### 8.1 SIGKILL動作の明確化

**重要**: Docker化はSIGKILLを「防ぐ」のではなく、「制御」することが目的です。

#### 実装前（ホスト実行）の問題
- 品質の悪いソルバーがホストのメモリを使い尽くす
- ホストOS全体がメモリ不足に陥る
- システム全体のOOM Killerが不規則に動作
- **制御不能なSIGKILL**: どのプロセスが終了されるか予測不可能

#### 実装後（Docker実行）の改善
- ソルバーはコンテナ内で隔離実行
- コンテナレベルでメモリ制限が設定
- **制御されたSIGKILL**: コンテナ内でのみ発生、ホスト保護
- プロセス終了の影響をコンテナ内に限定

#### 技術的実装詳細
```bash
docker run \
  --memory="16g" \
  --memory-swap="16g" \
  --oom-kill-disable=false  # false = コンテナ内でOOM Killerを有効化
```

**`--oom-kill-disable=false`の意味**:
- `true`: コンテナ内でOOM Killerを無効化（メモリ不足時にハング）
- `false`: コンテナ内でOOM Killerを有効化（メモリ不足時に適切に終了）

### 8.2 期待される動作
1. **正常ケース**: ソルバーがメモリ制限内で完了
2. **メモリ不足ケース**: 
   - コンテナのメモリ制限に達する
   - コンテナ内のOOM Killerが動作
   - プロセスが**コンテナ内で**SIGKILL終了
   - ホストシステムは影響を受けない
   - SolverResult.status = 'SIGKILL'として適切に記録

これにより、ホスト環境を保護しながら、問題のあるソルバーの動作を制御下で分析できます。

---

*最終更新: 2025年8月*