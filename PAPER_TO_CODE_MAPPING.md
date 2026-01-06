# BuPO 论文与代码映射文档

本文档详细说明了论文《Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies》中的核心概念在代码库中的具体实现位置。

---

## 📄 论文核心概念总览

论文提出了三个核心贡献：
1. **Internal Policy 分解**：将语言模型策略分解为内部层策略和模块策略
2. **Entropy 分析**：通过熵分析揭示不同模型的推理模式
3. **BuPO 算法**：自底向上的策略优化方法

---

## 1️⃣ Internal Policy 定义与实现

### 论文位置：Section 3.1 "Definition of Internal Policy"

#### 📖 论文公式

**Internal Layer Policy**（公式 6）：
```
π^l_Layer ≡ P^l_Layer = softmax(H^l E^T_u)
```

**Internal Modular Policy**（公式 7）：
```
π^l_ATTN = softmax(A^l E^T_u)
π^l_FFN = softmax(F^l E^T_u)
```

#### 💻 代码实现位置

**1. 自定义模型输出类**
- **文件**: `verl/models/custom_model/modeling_qwen3.py`
- **行号**: 88-120
- **类名**: `CausalLMOutputWithPastNew`
- **关键代码**:
```python
class CausalLMOutputWithPastNew(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[UserDict[str, Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mid_layer_logits: Optional[UserDict[int, torch.FloatTensor]] = None  # 存储内部层 logits
```

**2. 提取内部隐藏状态**
- **文件**: `verl/models/custom_model/modeling_qwen3.py`
- **行号**: 585-596
- **功能**: 从指定层提取隐藏状态并计算 logits
- **关键代码**:
```python
# 提取内部层隐藏状态
internal_logits = {}
startk = int(self.config.internal_layer)
for i in range(startk, startk+1):
    # H^l E^T_u 的实现
    internal_logits[i] = self.lm_head(outputs.hidden_states[i+1][:, slice_indices, :])
output.mid_layer_logits = internal_logits
```

**3. 内部策略前向传播**
- **文件**: `verl/workers/actor/dp_actor.py`
- **行号**: 357-555
- **方法**: `_forward_micro_batch_layer_k()`
- **功能**: 计算内部层策略的 log probability 和 entropy
- **关键代码**（行 460）:
```python
# 从内部层获取 logits
logits_rmpad = output.mid_layer_logits[layer_k].squeeze(0)
logits_rmpad.div_(temperature)
```

**4. 模型配置中的内部层设置**
- **文件**: `verl/models/custom_model/configuration_qwen3.py`
- **配置项**: `internal_layer`
- **说明**: 指定要优化的内部层索引（例如：layer 6）

---

## 2️⃣ Entropy 计算与分析

### 论文位置：Section 3.2 "Internal Policy Entropy Dynamics"

#### 📖 论文公式

**Internal Policy Entropy**（公式 8）：
```
H^l_Layer = -Σ P^l_Layer,j · log(P^l_Layer,j)
```

**Entropy Change**（公式 9）：
```
ΔH^l = H^l_Output - H^l_Input
```

#### 💻 代码实现位置

**1. Entropy 计算函数**
- **文件**: `verl/workers/actor/dp_actor.py`
- **相关行号**:
  - 行 351: `entropy = verl_F.entropy_from_logits(logits)`
  - 行 482: `entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)`
  - 行 549: 标准 entropy 计算

**2. Entropy 可视化分析**
- **文件**: `visualization/plot_internal_entropy.py`
- **行号**: 1-619（完整文件）
- **主要类**: `EntropyAnalyzer`
- **功能**:
  - 计算每层的 entropy（行 40-65）
  - 计算 entropy change（论文中的 ΔH）
  - 可视化 entropy 动态变化

**3. Hook 注册获取内部状态**
- **文件**: `visualization/plot_internal_entropy.py`
- **行号**: 66-115
- **方法**: `_register_hooks()`
- **功能**: 注册钩子函数获取 ATTN 和 FFN 的输入输出

**4. Entropy 计算实现**
- **文件**: `visualization/plot_internal_entropy.py`
- **行号**: 200-250（大约）
- **功能**: 对每个模块（Layer/ATTN/FFN）计算 entropy
- **实现**: 使用 `softmax(H^l E^T_u)` 然后计算 `-Σ p log(p)`

---

## 3️⃣ Internal Policy Optimization (InterGRPO)

### 论文位置：Section 4 "Internal Policy Optimization"

#### 📖 论文公式（公式 10）

```
J_InterGRPO(πθ, π^l_Layer) = E[min{r̂_i,t Â_i,t, clip(r̂_i,t, 1-ε, 1+ε)Â_i,t}]

其中: r̂_i,t = π^l_Layer(o_i,t|q,o_i,<t) / π^l_Layer,old(o_i,t|q,o_i,<t)
```

#### 💻 代码实现位置

**1. InterGRPO 实现的核心逻辑**
- **文件**: `verl/workers/actor/dp_actor.py`
- **行号**: 806-820（第一处调用）
- **关键判断**:
```python
# BuPO 模式判断
if self.config.internal_policy_interative:
    if micro_batch.meta_info['global_steps'] <= self.config.iterative_steps:
        # Phase 1: 优化内部层策略
        entropy, log_probs = self._forward_micro_batch_layer_k(
            model_inputs,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            layer_k=self.config.internal_layer
        )
    else:
        # Phase 2: 优化完整模型策略
        entropy, log_probs = self._forward_micro_batch(
            model_inputs,
            temperature=temperature,
            calculate_entropy=calculate_entropy
        )
```

**2. InterGRPO 实现的第二处**
- **文件**: `verl/workers/actor/dp_actor.py`
- **行号**: 906-920
- **说明**: 与第一处逻辑相同，处理不同的前向传播路径

**3. 梯度流控制**
- **文件**: Appendix A.5（论文）对应代码在训练时自动处理
- **原理**: 由于残差连接，优化 π^l_Layer 时只更新 layers 0 到 l 的参数
- **公式（论文公式 16）**:
```
∂J_InterGRPO/∂θ_k = {
    梯度计算, if k ≤ l
    0,        if k > l
}
```

---

## 4️⃣ Bottom-up Policy Optimization (BuPO) 算法

### 论文位置：Section 5 "Bottom-up Policy Optimization" + Algorithm 1

#### 📖 论文公式（公式 11）

```
J_BuPO(πθ, π^l_Layer) = {
    J_InterGRPO(πθ, π^l_Layer), if s_cur ≤ s_inter
    J_GRPO(πθ),                  if s_cur > s_inter
}
```

#### 💻 代码实现位置

**1. BuPO 配置参数**
- **文件**: `verl/workers/config/actor.py`
- **行号**: 225-240
- **配置类**: `FSDPActorConfig`
- **关键参数**:
```python
internal_policy_interative: bool = False  # 启用 BuPO
iterative_steps: int = 30                 # s_inter: 内部策略优化步数
internal_layer: int = 6                   # l: 要优化的层索引
```

**2. BuPO 配置文件**
- **文件**: `verl/trainer/config/actor/dp_actor.yaml`
- **行号**: 87-89
```yaml
internal_policy_interative: False
iterative_steps: 30
```

**3. BuPO 主训练入口**
- **文件**: `verl/trainer/main_ppo.py`
- **说明**: 使用 Hydra 配置系统，通过命令行参数覆盖配置
- **调用位置**: 训练脚本通过该文件启动 PPO 训练

**4. 自定义模型加载**
- **文件**: `verl/workers/fsdp_workers.py`
- **行号**: 249-256
- **功能**: 当启用 BuPO 时，替换标准 transformers 模型为自定义模型
- **关键代码**:
```python
if hasattr(self.config.actor, "internal_policy_interative") and \
   self.config.actor.internal_policy_interative:
    from verl.models.custom_model import modeling_qwen2 as custom_modeling_qwen2
    from verl.models.custom_model import modeling_qwen3 as custom_modeling_qwen3
    from verl.models.custom_model import modeling_llama as custom_modeling_llama
    sys.modules["transformers.models.qwen2.modeling_qwen2"] = custom_modeling_qwen2
    sys.modules["transformers.models.qwen3.modeling_qwen3"] = custom_modeling_qwen3
    sys.modules["transformers.models.llama.modeling_llama"] = custom_modeling_llama
```

---

## 5️⃣ 训练脚本与命令

### 论文位置：Section 5.1 "Main Results" - Training Setup

#### 💻 代码实现位置

**1. BuPO 训练脚本 (Qwen)**
- **文件**: `run_code/BuPO_qwen3.sh`
- **关键参数**（行 48-51, 102-104）:
```bash
k=5                                    # 内部层索引
iterative_steps=30                     # Phase 1 的训练步数
prompt_template_type="qwen3_no_thinking"
experiment_name="modelname_bupo_deepmath5k_${k}layerpolicy_iterstep${iterative_steps}..."

# Hydra 参数覆盖
actor_rollout_ref.actor.internal_policy_interative=True
actor_rollout_ref.actor.internal_layer=${k}
actor_rollout_ref.actor.iterative_steps=${iterative_steps}
actor_rollout_ref.model.override_config.internal_layer=${k}
```

**2. BuPO 训练脚本 (Llama)**
- **文件**: `run_code/BuPO_llama.sh`
- **说明**: 结构与 Qwen 版本相同，参数略有不同

**3. GRPO 基线脚本**
- **文件**: `run_code/GRPO_qwen3.sh`
- **说明**: 标准 GRPO，不启用 `internal_policy_interative`

**4. GRPO 基线脚本 (Llama)**
- **文件**: `run_code/GRPO_llama.sh`

---

## 6️⃣ 评估与可视化

### 论文位置：Section 5.1 "Main Results" - Evaluation Setup

#### 💻 代码实现位置

**1. 评估脚本**
- **文件**: `scripts/run_eval.sh`
- **行号**: 1-38
- **功能**: 在测试数据集上生成预测
- **关键配置**:
```bash
model=Qwen3-4B                        # 模型路径
dataset=math500                        # 数据集选择
tp_size=2                             # Tensor 并行大小
n_samples=1                           # 每个 prompt 的样本数
```

**2. 可视化脚本**
- **文件**: `visualization/plot_internal_entropy.py`
- **行号**: 完整文件（619行）
- **主要功能**:
  - 加载模型并注册 hooks（行 40-115）
  - 分析 entropy 动态（行 200-400）
  - 绘制 Figure 2, 3, 4（论文中的图）
  - 生成 Entropy Change 可视化

**3. 论文 Figure 对应**
- **Figure 2**: "Continuous Entropy Flow Through Layers"
  - 代码位置: `visualization/plot_internal_entropy.py` 中的绘图函数
  - 数据: Layer I/O, ATTN, FFN 的 entropy

- **Figure 3**: "Entropy Change Dynamics"
  - 代码位置: 计算 ΔH^l_ATTN 和 ΔH^l_FFN
  - 公式: ΔH^l = H^l_Output - H^l_Input

---

## 7️⃣ 数据流与训练流程

### 论文位置：Section 2.2 + Section 5

#### 📊 完整数据流

```
输入数据 (data/)
    ├── deepmath-5k.parquet          # 训练集
    ├── aime_2024.parquet            # 验证集
    ├── aime_2025.parquet
    ├── amc2023.parquet
    └── math500.parquet

                    ↓

verl/trainer/main_ppo.py             # 主训练入口
                    ↓

verl/workers/fsdp_workers.py         # 加载自定义模型
    (行 249-256)
                    ↓

verl/models/custom_model/            # 自定义模型
    ├── modeling_qwen3.py            # 返回 hidden_states 和 mid_layer_logits
    ├── modeling_llama.py
    └── configuration_*.py

                    ↓

verl/workers/actor/dp_actor.py       # Actor 实现
    ├── _forward_micro_batch_layer_k() (行 357)  # 内部层前向传播
    └── 两阶段切换逻辑 (行 806, 906)

                    ↓

训练输出
    └── checkpoints/BuPO/{experiment_name}/
```

#### 🔄 BuPO 两阶段训练流程

**Phase 1: Internal Policy Optimization** (步骤 1 到 `iterative_steps`)
1. 前向传播到第 k 层
2. 计算 π^k_Layer 的 log probs
3. 计算 importance ratio: r̂ = π^k / π^k_old
4. 使用 PPO loss 更新 layers 0 到 k
5. 梯度自动停止在第 k 层（残差连接特性）

**Phase 2: Full Model Optimization** (步骤 > `iterative_steps`)
1. 标准 GRPO/PPO 前向传播
2. 使用完整模型策略 πθ
3. 更新所有层参数

---

## 8️⃣ 关键超参数对应

### 论文位置：Table 5 (Appendix A.6)

| 论文参数 | 代码位置 | 默认值 | 说明 |
|---------|---------|--------|------|
| Policy learning rate | `actor_rollout_ref.actor.optim.lr` | 1e-6 | 策略网络学习率 |
| Training batch size | `data.train_batch_size` | 128 prompts | 每批 prompt 数量 |
| Samples per prompt | `actor_rollout_ref.rollout.n` | 8 | 每个 prompt 生成的响应数 |
| Mini-batch size | `actor_rollout_ref.actor.ppo_mini_batch_size` | 32 | PPO mini-batch |
| Max prompt length | `data.max_prompt_length` | 1024 tokens | 最大 prompt 长度 |
| Max response length | `data.max_response_length` | 7168 (Qwen) / 3072 (Llama) | 最大响应长度 |
| Rollout temperature | `actor_rollout_ref.rollout.temperature` | 1.0 | 采样温度 |
| Clip range ε | `actor_rollout_ref.actor.clip_ratio_low/high` | 0.2 | PPO 裁剪范围 |
| **BuPO specific** | | | |
| Internal layer k | `actor_rollout_ref.actor.internal_layer` | 5 (Qwen3-4B), 6 (Qwen3-8B) | 优化的内部层 |
| Iterative steps | `actor_rollout_ref.actor.iterative_steps` | 20-30 | Phase 1 步数 |

---

## 9️⃣ 实验结果对应

### 论文位置：Table 1 - Main Results

#### 📈 评估指标计算

**Avg@K**（论文中使用）:
- **实现**: 对每个问题生成 K 个响应，计算 Pass@1 的平均值
- **代码**: 评估逻辑在 vLLM 中处理

**Pass@K**（论文 Figure 7）:
- **公式**: `Pass@K = E[1 - C(n-c, K) / C(n, K)]`
- **参数**: n=300（总生成数），c（正确数），K（采样数）
- **范围**: K ∈ {1, 4, 16, 64, 256}

---

## 🔟 论文算法伪代码对应

### Algorithm 1: Bottom-up Policy Optimization (BuPO)

| 算法行 | 论文描述 | 代码位置 | 说明 |
|--------|---------|---------|------|
| Line 1 | Initialize scur ← 0 | 训练循环中的全局步数 | 由训练框架管理 |
| Line 3 | Sample batch q ~ Q | `data.train_files` | 从训练集采样 |
| Line 4 | Generate G outputs | `actor_rollout_ref.rollout.n=8` | 每个 prompt 生成 8 个响应 |
| Line 5 | Compute rewards and advantages | PPO 标准流程 | GRPO 优势估计 |
| Line 6 | if scur ≤ sinter | `verl/workers/actor/dp_actor.py:807` | 判断当前步数 |
| Line 7-9 | Phase 1: Optimize π^l_Layer | `_forward_micro_batch_layer_k()` | 内部策略优化 |
| Line 11-12 | Phase 2: Optimize πθ | `_forward_micro_batch()` | 完整模型优化 |
| Line 14 | Update parameters | 标准 PyTorch 优化器 | AdamW，lr=1e-6 |

---

## 📚 补充说明

### 代码中的命名约定

1. **Layer 索引**:
   - 论文中：Layer l ∈ [1, L]
   - 代码中：Layer index ∈ [0, L-1]
   - **注意**: 代码中的 layer 0 对应论文中的 layer 1

2. **Hidden states**:
   - `H^l` (论文) = `outputs.hidden_states[l+1]` (代码)
   - 因为 `hidden_states[0]` 是 embedding

3. **模块命名**:
   - ATTN (论文) = self_attn (代码)
   - FFN (论文) = mlp (代码)

### 关键文件总结

| 论文概念 | 主要实现文件 | 核心行号 |
|---------|------------|---------|
| Internal Policy 定义 | `verl/models/custom_model/modeling_qwen3.py` | 585-596 |
| Entropy 计算 | `verl/workers/actor/dp_actor.py` | 351, 482, 549 |
| BuPO 算法 | `verl/workers/actor/dp_actor.py` | 806-820, 906-920 |
| 训练配置 | `run_code/BuPO_qwen3.sh` | 48-51, 102-104 |
| 可视化分析 | `visualization/plot_internal_entropy.py` | 完整文件 |
| 评估脚本 | `scripts/run_eval.sh` | 完整文件 |

---

## ✅ 快速查找指南

**想要找论文中的某个概念？**

- **公式 6 (Internal Layer Policy)** → `verl/models/custom_model/modeling_qwen3.py:595`
- **公式 8 (Entropy 计算)** → `verl/workers/actor/dp_actor.py:351, 482`
- **公式 10 (InterGRPO)** → `verl/workers/actor/dp_actor.py:357-555`
- **公式 11 (BuPO)** → `verl/workers/actor/dp_actor.py:806-820`
- **Algorithm 1** → `verl/workers/actor/dp_actor.py` + 训练脚本
- **Figure 2 (Entropy Flow)** → `visualization/plot_internal_entropy.py`
- **Figure 3 (Entropy Change)** → `visualization/plot_internal_entropy.py`
- **Table 1 (Results)** → 通过 `scripts/run_eval.sh` 生成

---

## 🎯 运行完整流程

### 1. 训练 BuPO 模型
```bash
# 设置环境变量
export MODEL_PATH="your/model/path"
export DATA_PATH="your/data/path"

# 运行 BuPO 训练
bash run_code/BuPO_qwen3.sh
```

### 2. 评估模型
```bash
# 生成预测
bash scripts/run_eval.sh
```

### 3. 可视化分析
```bash
# 绘制 Internal Entropy 图
python visualization/plot_internal_entropy.py
```

---

**文档版本**: 1.0
**最后更新**: 2026-01-06
**对应论文**: arXiv:2512.19673v1
