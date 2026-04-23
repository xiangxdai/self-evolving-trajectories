# Cipher Datasets

本目录用于生成和管理 Cipher 任务数据集。

## 任务入口说明（重点）

- **主任务（main）**：`Anchored Global Dependency`  
  对应脚本：[`anchored_global_dependency.py`](./anchored_global_dependency.py)  
  这是当前主要使用、与正文主配置对齐的版本（默认 `Cipher-17`）。

- **补充变体（appendix）**：`Bi-directional Anchored Smoothing`  
  对应脚本：[`bidirectional_anchored_smoothing.py`](./bidirectional_anchored_smoothing.py)

## 1) 主任务：Anchored Global Dependency（主要）

脚本：`anchored_global_dependency.py`

默认配置：

- `n=17`
- `k_offset=5`
- `pos_const=[3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2]`
- `num_train=100000`
- `num_test=1000`

### 快速生成（默认 Cipher-17）

```bash
python data/cipher/anchored_global_dependency.py --out-dir data/cipher
```

默认输出文件名：

- `cipher17_anchored_global_mod10_train.jsonl`
- `cipher17_anchored_global_mod10_test.jsonl`

你也可以改前缀名：

```bash
python data/cipher/anchored_global_dependency.py \
  --out-dir data/cipher \
  --name cipher17_main
```

## 2) 补充变体：Bi-directional Anchored Smoothing

脚本：`bidirectional_anchored_smoothing.py`

默认配置：

- `n=9`
- `pos_const=[3,1,4,1,5,9,2,6,5]`

### 快速生成（默认 n=9）

```bash
python data/cipher/bidirectional_anchored_smoothing.py --out-dir data/cipher
```

### 生成 n=5 示例

```bash
python data/cipher/bidirectional_anchored_smoothing.py \
  --n 5 \
  --out-dir data/cipher \
  --name cipher5_bidir
```

## 数据格式

两个脚本都输出 JSONL，每行一个样本：

```json
{"input": "<ciphertext>", "output": "<plaintext>"}
```

- `input`：密文
- `output`：明文

## 备注

- 目录中的 `0main-en.pdf` 为主参考文档。  
- 当前代码组织约定：**主任务看 `anchored_global_dependency.py`，补充变体看 `bidirectional_anchored_smoothing.py`**。
