---
marp: true
theme: default
paginate: true
---

# Transformerを用いた時系列予測モデル
- 本プログラムは、Transformerアーキテクチャを使用して時系列データの予測を行うPythonプログラムです
- Encoder-Decoderモデルを実装し、過去のデータから将来の値を予測します

---

# プログラムの主要コンポーネント

1. **MyPositionalEncoding**
   - Transformerの位置エンコーディングを実装
   - 時系列データの位置情報を埋め込み

2. **MyTransformer**
   - メインのTransformerモデル
   - EncoderとDecoderの両方を含む
   - マルチヘッドアテンション機構を実装

---

# データ処理とモデルの構造

```mermaid
graph LR
    A[入力データ] --> B[特徴量スケーリング]
    B --> C[データセット作成]
    C --> D[Encoder]
    D --> E[Decoder]
    E --> F[予測出力]
```

- データは訓練用と検証用に分割
- 特徴量は正規化処理を実施
- バッチ処理に対応

---

# 学習プロセス

- **損失関数**: CrossEntropyLoss
- **最適化**: Adamオプティマイザー
- **評価指標**: RMSPE (Root Mean Square Percentage Error)
- **早期停止**: RMSPE値が閾値以下になった場合

```python
# 学習ループの基本構造
for epoch in range(100):
    model.train()
    # 訓練処理
    model.eval()
    # 検証処理
```

---

# 特徴と利点

1. **柔軟な時系列長**
   - 可変長の入力シーケンスに対応

2. **高度な並列処理**
   - マルチヘッドアテンションによる効率的な学習

3. **スケーラビリティ**
   - 様々なデータセットサイズに対応可能

4. **早期停止機能**
   - 効率的な学習プロセスの実現