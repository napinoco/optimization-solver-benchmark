# MATLAB統合 現行設計書

> **文書の目的**: この文書は、最適化ソルバーベンチマークシステムにおけるMATLAB関連コードコンポーネントの現在の機能とデータ処理フローを説明します。

---

## 目次

1. [アーキテクチャ概要](#アーキテクチャ概要)
2. [コアコンポーネント](#コアコンポーネント)
3. [データフローパイプライン](#データフローパイプライン)
4. [統合メカニズム](#統合メカニズム)
5. [問題解決システム](#問題解決システム)
6. [結果処理チェーン](#結果処理チェーン)
7. [エラーハンドリング戦略](#エラーハンドリング戦略)
8. [パフォーマンス考慮事項](#パフォーマンス考慮事項)

---

## アーキテクチャ概要

MATLAB統合は、MATLABの最適化ソルバー（SeDuMi、SDPT3）とPythonベースのベンチマークシステム間の完全なブリッジを提供します。アーキテクチャは明確な関心の分離を持つ**パイプライン設計**に従っています：

```
Pythonシステム ↔ matlab_solver.py ↔ MATLABスクリプト ↔ ソルバーバックエンド
```

### 主要設計原則

- **SolverInterface準拠**: Python `SolverInterface`仕様への完全な準拠
- **問題レジストリ統合**: YAML設定によるDIMACS/SDPLIB問題のシームレスな解決
- **フォルトトレランス**: すべてのレイヤーで意味のあるフォールバックを持つ包括的なエラーハンドリング
- **データ標準化**: PythonとMATLAB間の一貫したJSON交換フォーマット
- **一時ファイル管理**: 自動クリーンアップによる中間ファイルの安全な処理

---

## コアコンポーネント

### 1. Pythonレイヤー: `matlab_solver.py`

**目的**: MATLABソルバー用の`SolverInterface`を実装する主要なPythonインターフェース。

**主要クラス**:
- `MatlabSolver`: 完全なSolverInterface準拠を持つベースクラス
- `SeDuMiSolver`: SeDuMi用の便利ラッパー
- `SDPT3Solver`: SDPT3用の便利ラッパー

**コア機能**:
- **環境検証**: MATLAB/Octaveの可用性と起動パフォーマンスを検証
- **バージョン検出**: MATLABとソルバーバージョンの動的検出とキャッシュ
- **問題検証**: レジストリベースの問題互換性チェック
- **サブプロセス管理**: タイムアウト処理を持つ安全なMATLAB実行
- **一時ファイル管理**: 自動クリーンアップ付きのUUIDベース一時ファイル

### 2. MATLABオーケストレーター: `matlab_runner.m`

**目的**: すべてのユーティリティ関数を含む統合MATLABエントリーポイントで、問題読み込み、ソルバー実行、結果保存を調整。

**実行フロー**:
1. **設定読み込み**: 統合されたYAMLリーダーを使用して問題メタデータを取得
2. **問題読み込み**: ファイルタイプに基づいて適切なローダーにディスパッチ
3. **ソルバー実行**: 特定のソルバーランナー（SeDuMi/SDPT3）を呼び出し
4. **メトリクス計算**: 統合されたメトリクス計算機を使用して目的値と実行可能性測定値を計算
5. **結果シリアライゼーション**: 統合されたJSONフォーマッターとファイル保存機能を使用

**統合アーキテクチャ**: すべてのユーティリティ関数がメインオーケストレーター内にネストされた関数として埋め込まれ、外部依存関係を排除し、1,035行の単一ファイルへのデプロイメントを簡素化。

### 3. ソルバー実装

#### `sedumi_runner.m`
- **設定**: 公正なベンチマークのための最小限のソルバーオプション（サイレントモード用`fid=0`）
- **検証**: 包括的な入力検証と錐構造正規化
- **エラーハンドリング**: 構造化されたエラー結果作成による優雅な失敗
- **バージョン検出**: 多段階SeDuMiバージョン識別

#### `sdpt3_runner.m`
- **設定**: 詳細出力無効化によるデフォルトSDPT3パラメータ
- **前処理**: 入力検証と問題構造検証
- **結果マッピング**: SDPT3ステータスコードの標準形式への変換
- **パフォーマンス監視**: 解時間測定と反復回数カウント

### 4. 統合ユーティリティ関数

すべてのユーティリティ関数は、デプロイメントの簡素化と依存関係の削減のため、`matlab_runner.m`内にネストされた関数として統合されました：

#### コアユーティリティ（統合済み）
- **`calculate_solver_metrics()`**: 目的計算と実行可能性測定
- **`read_problem_registry()`**: 問題設定用のYAML解析  
- **`format_result_to_json()`**: 型安全性を持つMATLAB-JSON変換
- **`save_json_safely()`**: エラー回復を持つ堅牢なファイルI/O
- **`save_solutions_if_needed()`**: オプションの解ベクトル永続化
- **`detect_versions()`**: 包括的な環境とソルバーバージョン検出

#### サポート関数（統合済み）
- **`calculate_dual_cone_violation()`**: 錐固有の双対非実行可能性計算
- **`proj_onto_soc()`**: 二次錐射影
- **`convert_to_json_compatible()`**: JSONシリアライゼーション用データ型変換
- **`get_field_safe()`**: デフォルト付き安全な構造体フィールドアクセス
- **`detect_sedumi_version()`** / **`detect_sdpt3_version()`**: ソルバー固有バージョン検出

**統合の利点**:
- **単一ファイルデプロイメント**: すべてのMATLAB機能が一つのファイルに
- **パス依存関係の削減**: 複数のスクリプトパス管理が不要  
- **アトミック操作**: 外部依存関係なしでアクセス可能なすべての機能
- **簡素化されたメンテナンス**: 更新とバージョン管理が単一ファイルで完結

---

## データフローパイプライン

### フェーズ1: 初期化と検証

```
Pythonリクエスト → 問題データ検証 → レジストリ検索 → ソルバー互換性チェック
```

1. **Pythonレイヤー**: `ProblemData`構造と必要フィールドを検証
2. **レジストリ解決**: `problem_registry.yaml`で問題を検索してファイルパスとタイプを取得
3. **互換性チェック**: ソルバーが問題タイプ（MAT/DAT-Sファイル）をサポートすることを検証
4. **環境チェック**: MATLABの可用性とソルバーインストールを確認

### フェーズ2: MATLAB実行

```
一時ファイル作成 → MATLABコマンド構築 → サブプロセス実行
```

1. **一時ファイル管理**: UUIDベースのJSON結果ファイルを作成
2. **コマンド構築**: 適切なパス処理で安全なMATLABコマンドを構築:
   ```matlab
   addpath('matlab_script_dir'); matlab_runner('problem_name', 'solver_name', 'result_file')
   ```
3. **サブプロセス実行**: タイムアウト付きでMATLABを実行し、stdout/stderrをキャプチャ

### フェーズ3: 問題解決と読み込み

```
問題レジストリ → ファイルパス解決 → 形式固有読み込み
```

1. **レジストリ解析**: 統合された`read_problem_registry()`が問題メタデータを抽出
2. **パス解決**: 相対パスを絶対パスに変換
3. **形式ディスパッチ**: ファイルタイプに基づいて`mat_loader()`または`dat_loader()`にルーティング
4. **データ抽出**: 行列`A`、`b`、`c`と錐構造`K`を返す

### フェーズ4: ソルバー実行

```
入力検証 → ソルバー設定 → 問題解決 → ステータスマッピング
```

1. **入力検証**: 行列次元と錐構造の一貫性をチェック
2. **ソルバー設定**: 公正なベンチマークのための最小限の設定を適用
3. **解決プロセス**: タイミング測定付きで最適化を実行
4. **結果処理**: ソルバー固有のステータスコードを標準形式にマップ

### フェーズ5: メトリクス計算

```
解ベクトル → 目的計算 → 実行可能性評価 → ギャップ計算
```

1. **目的値**（統合された`calculate_solver_metrics()`経由）: 
   - 主問題: `c' * x`
   - 双対問題: `b' * y`
2. **非実行可能性測定**:
   - 主問題: `||A*x - b|| / (1 + ||b||)`
   - 双対問題: `sqrt(cone_violation(c - A'*y)) / (1 + ||c||^2)`
3. **双対性ギャップ**: `|primal_objective - dual_objective|`

### フェーズ6: 結果シリアライゼーションと返却

```
MATLAB結果 → JSON変換 → ファイル書き込み → Python読み取り → SolverResult作成
```

1. **JSON形式化**: 統合された`format_result_to_json()`がMATLAB構造体をJSON文字列に変換
2. **ファイル書き込み**: 統合された`save_json_safely()`がエラー回復付きでJSONを書き込み
3. **Python読み取り**: `json.load()`が結果ファイルを解析
4. **結果変換**: メタデータ拡張付きで`SolverResult`オブジェクトにマップ

---

## 統合メカニズム

### 問題レジストリ統合

システムは問題名をファイルの場所にマップするために集中化されたYAML設定を使用：

```yaml
problem_libraries:
  nb:
    display_name: "Network Design"
    file_path: "problems/DIMACS/data/ANTENNA/nb.mat.gz"
    file_type: "mat"
    library_name: "DIMACS"
  
  arch0:
    display_name: "Architecture 0"
    file_path: "problems/SDPLIB/data/arch0.dat-s"
    file_type: "dat-s"
    library_name: "SDPLIB"
```

**解決プロセス**:
1. Python `ProblemData.name` → レジストリ検索
2. `file_path`と`file_type`を抽出
3. 形式固有読み込みのためにMATLABに渡す
4. 適切なローダーを使用して問題データを読み込み

### 一時ファイル管理

**UUIDベースネーミング**: 各ソルバー実行は一意の一時ファイルを作成：
```
/tmp/matlab_sedumi_result_<uuid>.json
```

**自動クリーンアップ**: コンテキストマネージャーが例外時でもクリーンアップを保証：
```python
with temp_file_context(".json") as result_file:
    # MATLABソルバーを実行
    # ファイルは自動的にクリーンアップされる
```

**孤児検出**: 1時間以上古い放置された一時ファイルの定期的なクリーンアップ。

### バージョン検出とキャッシュ

**多段階検出**:
1. **関数存在**: ソルバー関数が利用可能かチェック
2. **バージョン関数**: ソルバー固有のバージョン関数を呼び出し
3. **ファイル解析**: インストールファイルからバージョンを解析
4. **機能テスト**: 最小問題でソルバーが動作することを検証

**キャッシュ戦略**: 繰り返し検出オーバーヘッドを避けるために初期化時にバージョン情報をキャッシュ。

---

## 結果処理チェーン

### MATLAB結果構造

```matlab
result = struct();
result.solver_name = 'SeDuMi';
result.solver_version = 'SeDuMi-1.3.7';
result.status = 'optimal';
result.solve_time = 0.245;
result.primal_objective = -4.567;
result.dual_objective = -4.567;
result.gap = 1.234e-10;
result.primal_infeasibility = 2.345e-12;
result.dual_infeasibility = 3.456e-11;
result.iterations = 15;
```

### JSON変換プロセス

**型安全性**: 統合された`format_result_to_json()`がMATLAB固有型を処理：
- `NaN` → `null`（JSONで空配列）
- `±Inf` → `±1e308` 
- 空配列 → `null`
- 数値精度保持を保証

### Python SolverResultマッピング

```python
SolverResult(
    solve_time=matlab_result['solve_time'],
    status=matlab_result['status'].upper(),
    primal_objective_value=safe_float(matlab_result['primal_objective_value']),
    dual_objective_value=safe_float(matlab_result['dual_objective_value']),
    duality_gap=safe_float(matlab_result['duality_gap']),
    primal_infeasibility=safe_float(matlab_result['primal_infeasibility']),
    dual_infeasibility=safe_float(matlab_result['dual_infeasibility']),
    iterations=safe_int(matlab_result['iterations']),
    solver_name='matlab_sedumi',
    solver_version='SeDuMi-1.3.7 (MATLAB R2023b)',
    additional_info={
        'matlab_output': matlab_result,
        'solver_backend': 'sedumi',
        'execution_environment': 'matlab'
    }
)
```

---

## エラーハンドリング戦略

### 多層エラー回復

#### 1. Pythonレイヤー
- **サブプロセス失敗**: 診断情報のためにstderr/stdoutをキャプチャ
- **タイムアウト処理**: 構造化されたタイムアウト結果を返す
- **ファイルI/Oエラー**: 欠損/破損結果ファイルを処理
- **JSON解析**: 不正なJSONの優雅な処理

#### 2. MATLABレイヤー
- **ソルバーエラー**: 診断情報付きエラー結果構造を作成
- **問題読み込み**: 欠損ファイルまたは形式エラーを処理
- **メトリクス計算**: 計算でのNaN/Infの安全な処理
- **ファイル書き込み**: 失敗時のロールバック付きアトミック操作

### エラー結果構造

```python
SolverResult.create_error_result(
    error_message="MATLAB実行失敗: ソルバーが見つかりません",
    solve_time=actual_time_spent,
    solver_name=self.solver_name,
    solver_version=self.get_version()
)
```

### 診断情報

すべてのエラー結果には以下が含まれます：
- **エラー分類**: タイムアウト、ソルバーエラー、I/Oエラーなど
- **実行コンテキスト**: MATLABバージョン、ソルバー可用性、ファイルパス
- **タイミング情報**: 失敗前に費やした時間
- **生出力**: デバッグ用のMATLAB stdout/stderr

---

## パフォーマンス考慮事項

### 起動最適化

**MATLAB初期化**: コールドMATLAB起動は5-15秒かかる可能性
- **タイムアウト調整**: 起動用に解タイムアウトに15秒バッファを追加
- **起動警告**: MATLAB起動が10秒以上の場合にアラート
- **事前ウォーミング**: 本番環境では事前ウォーミングされたMATLABセッションを検討

### メモリ管理

**一時ファイル**: 
- UUIDベースネーミングが競合を防止
- 自動クリーンアップがディスク容量問題を防止
- 孤児検出が中断された実行を処理

**MATLABメモリ**:
- 使用後に大きな行列を解放
- MATLABワークスペースの一時変数をクリア
- 長時間実行セッションでメモリ使用量を監視

### 実行効率

**パス管理**: 
- セッションごとに一度MATLABスクリプトパスを追加
- 作業ディレクトリ問題を避けるために絶対パスを使用
- 繰り返しYAML解析を避けるために問題レジストリをキャッシュ

**サブプロセス最適化**:
- 解決ごとに単一MATLABコマンド（複数呼び出しなし）
- 可能な場合は複数問題をバッチ処理
- データシリアライゼーションオーバーヘッドを最小化

---

## 統合ポイント要約

### Pythonベンチマークシステムとの統合

1. **SolverInterface準拠**: 他のソルバーのドロップイン置換
2. **問題レジストリ**: シームレスなDIMACS/SDPLIB問題解決  
3. **データベース統合**: ストレージ用の標準SolverResult形式
4. **レポート生成**: レポートパイプライン用の一貫したメタデータ

### 外部ライブラリとの統合

1. **DIMACS問題**: `mat_loader.m`によるネイティブMATファイルサポート
2. **SDPLIB問題**: `dat_loader.m`によるDAT-Sファイルサポート
3. **問題メタデータ**: レジストリ駆動の問題分類とメタデータ

### MATLABエコシステムとの統合

1. **ソルバー検出**: インストールされたソルバーの動的発見
2. **バージョン追跡**: 包括的なバージョンメタデータ収集
3. **エラー伝播**: MATLABからPythonへの意味のあるエラーメッセージ
4. **解ストレージ**: 詳細分析用のオプションMATファイル出力

この設計は、既存のPythonアーキテクチャ内で包括的なMATLABソルバーサポートを提供しながら、公正なベンチマーク哲学を維持する堅牢で本番対応の統合を提供します。