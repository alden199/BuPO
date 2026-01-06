# BuPO 算法详细实现解析

本文档逐行解释 BuPO（Bottom-up Policy Optimization）算法的完整实现细节，面向不熟悉 verl 框架的读者。

---

## 📚 目录

1. [verl 框架基础概念](#1-verl-框架基础概念)
2. [BuPO 训练流程总览](#2-bupo-训练流程总览)
3. [训练入口：main_ppo.py](#3-训练入口mainppopy)
4. [Actor 核心实现：dp_actor.py](#4-actor-核心实现dpactorpy)
5. [BuPO 两阶段切换逻辑](#5-bupo-两阶段切换逻辑)
6. [内部层前向传播](#6-内部层前向传播)
7. [自定义模型实现](#7-自定义模型实现)
8. [Loss 计算与反向传播](#8-loss-计算与反向传播)
9. [完整训练循环](#9-完整训练循环)
10. [配置参数详解](#10-配置参数详解)

---

## 1. verl 框架基础概念

### 1.1 verl 是什么？

**verl (Volcano Engine Reinforcement Learning)** 是字节跳动开源的大模型强化学习训练框架，专门为 LLM 的 RL 训练设计。

### 1.2 核心组件

```
verl 框架架构
├── Trainer (训练器)
│   ├── main_ppo.py           # 主训练入口
│   └── RayPPOTrainer         # Ray 分布式训练器
│
├── Workers (工作节点)
│   ├── Actor (策略网络)       # 生成动作并更新策略
│   ├── Critic (价值网络)      # 估计状态价值（PPO 需要）
│   ├── Rollout (推理引擎)     # 使用 vLLM 生成响应
│   └── Reference (参考策略)   # 用于 KL 惩罚
│
├── Models (模型层)
│   ├── custom_model/         # 自定义模型（支持内部层输出）
│   └── transformers 标准模型
│
└── Data (数据层)
    └── DataProto              # 统一数据格式
```

### 1.3 关键概念

#### 1.3.1 DataProto

verl 使用 `DataProto` 统一管理数据，包含：
- **batch**: 张量数据（input_ids, attention_mask 等）
- **non_tensor_batch**: 非张量数据（图像等）
- **meta_info**: 元信息（global_steps, temperature 等）

#### 1.3.2 Actor vs Rollout

- **Rollout**: 使用 vLLM 快速生成多个响应（推理优化）
- **Actor**: 使用完整模型计算 log_probs 和梯度（训练优化）

#### 1.3.3 FSDP (Fully Sharded Data Parallel)

PyTorch 的分布式训练策略，将模型参数、梯度、优化器状态分片到多个 GPU。

---

## 2. BuPO 训练流程总览

### 2.1 整体流程图

```
用户启动训练
    ↓
bash run_code/BuPO_qwen3.sh
    ↓
python -m verl.trainer.main_ppo (Hydra 配置)
    ↓
main() 函数 → run_ppo()
    ↓
初始化 Ray 集群
    ↓
创建 TaskRunner (远程执行)
    ↓
TaskRunner.run()
    ↓
创建 RayPPOTrainer
    ↓
┌─────────────────────────────────────┐
│   主训练循环 (每个 step)              │
├─────────────────────────────────────┤
│  1. Rollout: 生成响应 (vLLM)         │
│  2. 计算 Reward                      │
│  3. 计算 Advantage (GRPO)           │
│  4. Actor Update (BuPO 核心)        │
│     ├─ 判断当前步数                  │
│     ├─ Phase 1: 优化内部层策略       │
│     └─ Phase 2: 优化完整模型         │
│  5. Reference Policy Update         │
│  6. 保存 Checkpoint                 │
└─────────────────────────────────────┘
```

### 2.2 BuPO 特有的训练阶段

**Phase 1: Internal Policy Optimization**
- 训练步数: 1 到 `iterative_steps`（例如 30）
- 优化目标: π^k_Layer（第 k 层的内部策略）
- 更新范围: 只更新 layers 0 到 k

**Phase 2: Full Model Optimization**
- 训练步数: `iterative_steps + 1` 到 `total_steps`
- 优化目标: πθ（完整模型策略）
- 更新范围: 更新所有层

---

## 3. 训练入口：main_ppo.py

### 3.1 文件位置
```
verl/trainer/main_ppo.py
```

### 3.2 main() 函数（第 35-42 行）

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)
```

**逐行解析**:

**第 35 行**: `@hydra.main(...)`
- **Hydra 装饰器**: 自动加载和管理配置文件
- `config_path="config"`: 配置文件所在目录
- `config_name="ppo_trainer"`: 主配置文件名
- **作用**: 将 YAML 配置和命令行参数合并成 `config` 对象

**示例**: 当你运行训练脚本时
```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.internal_policy_interative=True \
    actor_rollout_ref.actor.internal_layer=5
```
Hydra 会：
1. 加载 `config/ppo_trainer.yaml`
2. 用命令行参数覆盖配置
3. 生成最终的 `config` 对象

### 3.3 run_ppo() 函数（第 46-91 行）

```python
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process."""

    # 第 55-66 行: 初始化 Ray
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
```

**逐行解析**:

**第 55 行**: `if not ray.is_initialized():`
- **检查**: Ray 是否已经初始化（避免重复初始化）

**第 60 行**: `default_runtime_env = get_ppo_ray_runtime_env()`
- **获取默认运行环境**: 设置环境变量（TOKENIZERS_PARALLELISM, NCCL_DEBUG 等）

**第 61-64 行**: 合并配置
- 将用户自定义的 Ray 配置和默认配置合并

**第 66 行**: `ray.init(...)`
- **初始化 Ray 集群**:
  - 单机多卡: Ray 管理本地 GPU
  - 多机多卡: Ray 连接到远程集群

```python
    # 第 84-85 行: 创建远程任务执行器
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
```

**逐行解析**:

**第 84 行**: `runner = TaskRunner.remote()`
- **创建 Ray Actor**: `TaskRunner` 类被 `@ray.remote` 装饰
- **remote()**: 在 Ray 集群中创建一个远程实例
- **作用**: 任务会在分配的 CPU 核心上运行（不占用 GPU）

**第 85 行**: `ray.get(runner.run.remote(config))`
- `runner.run.remote(config)`: 远程调用 `run` 方法（异步）
- `ray.get(...)`: 等待远程调用完成并获取返回值
- **作用**: 阻塞主进程直到训练完成

### 3.4 TaskRunner 类（第 94-100+ 行）

```python
@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks."""

    def run(self, config):
        # ... 创建 RayPPOTrainer 并启动训练 ...
```

**关键点**:
- `@ray.remote(num_cpus=1)`: 分配 1 个 CPU 核心
- `run()` 方法内部创建 `RayPPOTrainer` 并调用其 `train()` 方法

---

## 4. Actor 核心实现：dp_actor.py

### 4.1 文件位置
```
verl/workers/actor/dp_actor.py
总行数: 975 行
```

### 4.2 DataParallelPPOActor 类

这是 BuPO 的核心实现，包含：
- 策略网络的前向传播
- Loss 计算
- 梯度更新
- **BuPO 特有的两阶段切换逻辑**

### 4.3 类初始化（第 52-150 行，大致）

```python
class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer,
        actor_scheduler,
    ):
        self.config = config
        self.actor_module = actor_module  # FSDP 包装的模型
        self.actor_optimizer = actor_optimizer
        self.actor_scheduler = actor_scheduler

        # BuPO 相关配置
        # config.internal_policy_interative: 是否启用 BuPO
        # config.internal_layer: 优化哪一层（例如 5）
        # config.iterative_steps: Phase 1 的步数（例如 30）
```

**关键属性**:
- `self.config`: Actor 配置（包含 BuPO 参数）
- `self.actor_module`: 经过 FSDP 包装的模型
- `self.use_remove_padding`: 是否使用 packed attention（Flash Attention）
- `self.use_ulysses_sp`: 是否使用 Ulysses 序列并行

---

## 5. BuPO 两阶段切换逻辑

### 5.1 compute_ref_log_prob() 中的切换（第 806-819 行）

```python
# 文件: verl/workers/actor/dp_actor.py
# 方法: compute_ref_log_prob()
# 位置: 第 806-819 行

for micro_batch in micro_batches:
    micro_batch = micro_batch.to(get_device_id())
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

    with torch.no_grad():  # 不计算梯度（推理模式）
        # ============ BuPO 核心逻辑 ============
        if self.config.internal_policy_interative:  # 第 806 行
            # 判断当前是 Phase 1 还是 Phase 2
            if micro_batch.meta_info['global_steps'] <= self.config.iterative_steps:  # 第 807 行
                # Phase 1: 使用内部层策略
                entropy, log_probs = self._forward_micro_batch_layer_k(  # 第 808 行
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    layer_k=self.config.internal_layer  # 例如 layer_k=5
                )
            else:
                # Phase 2: 使用完整模型策略
                entropy, log_probs = self._forward_micro_batch(  # 第 812 行
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy
                )
        # ============ 标准 GRPO 逻辑 ============
        else:
            entropy, log_probs = self._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                calculate_entropy=calculate_entropy
            )
```

### 5.2 逐行详细解析

#### 第 806 行: `if self.config.internal_policy_interative:`

**问题**: 什么是 `internal_policy_interative`？

**答案**:
- 这是一个**布尔配置参数**，控制是否启用 BuPO
- 在训练脚本中设置:
  ```bash
  actor_rollout_ref.actor.internal_policy_interative=True
  ```
- **True**: 启用 BuPO（两阶段训练）
- **False**: 标准 GRPO（始终优化完整模型）

#### 第 807 行: `if micro_batch.meta_info['global_steps'] <= self.config.iterative_steps:`

**问题**: `global_steps` 和 `iterative_steps` 是什么？

**答案**:
- **`global_steps`**: 当前训练的全局步数（1, 2, 3, ..., total_steps）
- **`iterative_steps`**: Phase 1 的持续步数（例如 30）
- **判断逻辑**:
  - `global_steps <= iterative_steps` (例如 1-30): Phase 1
  - `global_steps > iterative_steps` (例如 31-300): Phase 2

**示例**:
```python
# 假设 iterative_steps = 30, total_steps = 300
# global_steps = 1:  Phase 1 (优化内部层)
# global_steps = 30: Phase 1 (优化内部层)
# global_steps = 31: Phase 2 (优化完整模型)
# global_steps = 300: Phase 2 (优化完整模型)
```

#### 第 808-810 行: `self._forward_micro_batch_layer_k(...)`

**问题**: 这个函数做什么？

**答案**:
- **Phase 1 的核心函数**: 计算内部层策略的 log_probs
- **输入**:
  - `model_inputs`: 包含 input_ids, attention_mask 等
  - `layer_k`: 要优化的内部层索引（例如 5）
  - `temperature`: 采样温度（例如 1.0）
- **输出**:
  - `log_probs`: 内部层策略的对数概率
  - `entropy`: 策略的熵（可选）

**关键**: 它会调用**自定义模型**，获取第 k 层的隐藏状态

#### 第 812-814 行: `self._forward_micro_batch(...)`

**问题**: 这和上面有什么区别？

**答案**:
- **Phase 2 和标准 GRPO 使用的函数**: 计算完整模型策略的 log_probs
- **区别**:
  | 特性 | `_forward_micro_batch_layer_k` | `_forward_micro_batch` |
  |------|-------------------------------|------------------------|
  | 使用阶段 | Phase 1 (BuPO) | Phase 2 + 标准 GRPO |
  | 计算层 | 第 k 层 | 最后一层 |
  | 输出 | π^k_Layer(a\|s) | πθ(a\|s) |
  | 梯度更新 | 只更新 layers 0-k | 更新所有层 |

### 5.3 update_policy() 中的切换（第 906-918 行）

**这是第二处相同的切换逻辑**，在策略更新时使用：

```python
# 文件: verl/workers/actor/dp_actor.py
# 方法: update_policy()
# 位置: 第 906-918 行

calculate_entropy = False
if entropy_coeff != 0:
    calculate_entropy = True

# ============ BuPO 核心逻辑（与上面完全相同）============
if self.config.internal_policy_interative:  # 第 906 行
    if micro_batch.meta_info['global_steps'] <= self.config.iterative_steps:  # 第 907 行
        # Phase 1
        entropy, log_prob = self._forward_micro_batch_layer_k(  # 第 908 行
           model_inputs,
           temperature=temperature,
           calculate_entropy=calculate_entropy,
           layer_k=self.config.internal_layer
        )
    else:
        # Phase 2
        entropy, log_prob = self._forward_micro_batch(  # 第 912 行
            model_inputs,
            temperature=temperature,
            calculate_entropy=calculate_entropy
        )
else:
    # 标准 GRPO
    entropy, log_prob = self._forward_micro_batch(
        model_inputs,
        temperature=temperature,
        calculate_entropy=calculate_entropy
    )
```

**为什么有两处？**

1. **第一处**（`compute_ref_log_prob`）: 用于计算参考策略的 log_probs（推理模式，不计算梯度）
2. **第二处**（`update_policy`）: 用于策略更新（训练模式，计算梯度）

---

## 6. 内部层前向传播

### 6.1 _forward_micro_batch_layer_k() 方法

```
文件: verl/workers/actor/dp_actor.py
方法: _forward_micro_batch_layer_k()
位置: 第 357-555 行（约 200 行）
```

这是 **BuPO 最核心的函数**，实现了内部层策略的前向传播。

### 6.2 函数签名（第 357-359 行）

```python
def _forward_micro_batch_layer_k(
    self,
    micro_batch,
    temperature,
    calculate_entropy=False,
    layer_k=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        entropy: # (bs, response_len)
        log_probs: # (bs, response_len)
    """
```

**参数说明**:
- `micro_batch`: 一个小批次的数据
- `temperature`: 采样温度（控制分布的平滑程度）
- `calculate_entropy`: 是否计算熵
- `layer_k`: 内部层索引（例如 5）

**返回值**:
- `entropy`: 策略熵，shape=(batch_size, response_length)
- `log_probs`: 对数概率，shape=(batch_size, response_length)

### 6.3 获取输入数据（第 366-387 行）

```python
response_length = micro_batch["responses"].size(-1)

# 处理多模态输入（图像等）
multi_modal_inputs = {}
if "multi_modal_inputs" in micro_batch.keys():
    # ... 处理图像输入 ...

with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
    input_ids = micro_batch["input_ids"]  # (batch_size, total_length)
    batch_size, seqlen = input_ids.shape
    attention_mask = micro_batch["attention_mask"]  # (batch_size, total_length)
    position_ids = micro_batch["position_ids"]  # (batch_size, total_length)
    entropy = None
```

**逐行解析**:

**第 366 行**: `response_length = micro_batch["responses"].size(-1)`
- **responses**: 模型生成的响应部分
- **作用**: 后续只计算响应部分的 log_probs（不包括 prompt）

**第 379 行**: `with torch.autocast(..., dtype=torch.bfloat16):`
- **混合精度训练**: 使用 bfloat16 加速计算
- **自动类型转换**: PyTorch 自动在 bfloat16 和 float32 之间切换

**第 380-383 行**: 提取输入
- `input_ids`: Token IDs，形状 (batch_size, total_length)
  - `total_length = prompt_length + response_length`
- `attention_mask`: 注意力掩码，1 表示有效 token，0 表示填充
- `position_ids`: 位置编码

### 6.4 Packed Attention 处理（第 388-438 行）

**什么是 Packed Attention？**

标准 Attention:
```
Batch 1: [token1, token2, token3, PAD, PAD]
Batch 2: [token1, PAD, PAD, PAD, PAD]
Batch 3: [token1, token2, token3, token4, PAD]
```

Packed Attention (Flash Attention):
```
Packed: [batch1_token1, batch1_token2, batch1_token3, batch2_token1, batch3_token1, ...]
         ^                                              ^             ^
         cu_seqlens[0]                                  cu_seqlens[1] cu_seqlens[2]
```

**优点**:
- 去除 PAD token，减少计算量
- 支持变长序列，提高效率

```python
if self.use_remove_padding:  # 第 388 行
    # 去除 padding，变成 packed 格式
    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
        input_ids.unsqueeze(-1), attention_mask
    )  # input_ids_rmpad shape: (total_nnz, 1)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

    # 同样处理 position_ids
    position_ids_rmpad = index_first_axis(
        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
        indices
    ).transpose(0, 1)
```

**逐行解析**:

**第 388 行**: `if self.use_remove_padding:`
- **判断**: 是否使用 packed attention（推荐，更快）

**第 389-391 行**: `unpad_input(...)`
- **Flash Attention 工具**: 去除 padding tokens
- **输入**: `input_ids` + `attention_mask`
- **输出**:
  - `input_ids_rmpad`: 压缩后的 token IDs
  - `indices`: 有效 token 的索引
  - `cu_seqlens`: 累积序列长度（用于分隔不同样本）
    ```python
    # 示例
    cu_seqlens = [0, 3, 4, 8]  # 表示:
    # Batch 0: tokens 0-2 (3 个 tokens)
    # Batch 1: tokens 3-3 (1 个 token)
    # Batch 2: tokens 4-7 (4 个 tokens)
    ```

**第 414 行**: `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`
- **作用**: 将序列向左移动一位，用于计算 log_probs
- **为什么？** 因为要计算 P(token_t | context<t)
  ```python
  原始: [A, B, C, D]
  rolled: [B, C, D, A]  # 用于匹配 labels
  ```

### 6.5 调用模型（第 446-454 行）

```python
output = self.actor_module(
    input_ids=input_ids_rmpad,  # (1, total_nnz)
    attention_mask=None,  # packed attention 不需要
    position_ids=position_ids_rmpad,
    **multi_modal_inputs,
    use_cache=False,  # 训练时不缓存 KV
    output_hidden_states=True,  # ★ 关键: 输出所有层的隐藏状态
    **extra_args,
)
```

**逐行解析**:

**第 446-454 行**: 模型前向传播
- **input_ids**: 压缩后的输入
- **attention_mask=None**: packed attention 通过 `cu_seqlens` 处理
- **output_hidden_states=True**: ⭐ **BuPO 的关键**
  - 让模型返回**所有层**的隐藏状态
  - 标准模型只返回最后一层
  - 自定义模型会返回 `hidden_states[0], hidden_states[1], ..., hidden_states[L]`

**返回的 output 对象包含**:
- `output.logits`: 最后一层的 logits (vocabulary distribution)
- `output.hidden_states`: 所有层的隐藏状态（tuple）
- `output.mid_layer_logits`: ⭐ **BuPO 添加的字段**
  - 自定义模型计算的内部层 logits
  - `mid_layer_logits[k]` = 第 k 层的 logits

### 6.6 提取内部层 logits（第 458-462 行）

```python
if self.use_fused_kernels:  # 第 455 行
    # 使用融合内核（不常用）
    log_probs = output.log_probs.squeeze(0)
    entropy_rmpad = output.entropy.squeeze(0)
else:  # 第 458 行 - 常用路径
    # ★★★ 关键: 从自定义模型获取第 k 层的 logits ★★★
    logits_rmpad = output.mid_layer_logits[layer_k].squeeze(0)  # 第 460 行
    logits_rmpad.div_(temperature)  # 第 461 行
```

**逐行解析**:

**第 460 行**: `logits_rmpad = output.mid_layer_logits[layer_k].squeeze(0)`
- **mid_layer_logits**: 自定义模型返回的字典
  - `mid_layer_logits[5]`: 第 5 层的 logits
  - Shape: (1, total_nnz, vocab_size)
- **squeeze(0)**: 去掉第一个维度
  - Shape 变为: (total_nnz, vocab_size)

**问题**: `mid_layer_logits` 是怎么计算的？

**答案**: 在自定义模型中（稍后详解）：
```python
# verl/models/custom_model/modeling_qwen3.py: 行 593-596
startk = int(self.config.internal_layer)  # 例如 5
for i in range(startk, startk+1):
    # H^k E^T_u (论文公式 6)
    internal_logits[i] = self.lm_head(outputs.hidden_states[i+1])
```

**第 461 行**: `logits_rmpad.div_(temperature)`
- **温度缩放**: logits / temperature
- **作用**: 控制分布的平滑程度
  - temperature = 1.0: 不变
  - temperature > 1.0: 更平滑（熵增加）
  - temperature < 1.0: 更尖锐（熵减少）

### 6.7 计算 Log Probs（第 466-470 行）

```python
log_probs = logprobs_from_logits(
    logits=logits_rmpad,  # (total_nnz, vocab_size)
    labels=input_ids_rmpad_rolled,  # (total_nnz,) - rolled labels
    inplace_backward=inplace_backward,  # 是否原地梯度
)
```

**逐行解析**:

**logprobs_from_logits() 函数**:
```python
# 实现逻辑（简化版）
def logprobs_from_logits(logits, labels, inplace_backward):
    # 1. 计算 log softmax
    log_probs_all = F.log_softmax(logits, dim=-1)  # (total_nnz, vocab_size)

    # 2. 提取 labels 对应的 log_probs
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1))

    # 3. 去掉最后一维
    log_probs = log_probs.squeeze(-1)  # (total_nnz,)

    return log_probs
```

**示例**:
```python
# 假设 vocab_size = 3, total_nnz = 4
logits = [[2.0, 1.0, 0.5],  # token 0 的 logits
          [1.5, 2.5, 1.0],  # token 1 的 logits
          [0.5, 1.0, 2.0],  # token 2 的 logits
          [2.0, 2.0, 2.0]]  # token 3 的 logits

labels = [1, 2, 2, 0]  # 实际生成的 token IDs

# log_softmax 后
log_probs_all = [[-0.5, -1.5, -2.0],
                 [-1.5, -0.5, -2.0],
                 [-2.5, -2.0, -1.0],
                 [-1.1, -1.1, -1.1]]

# gather 提取对应 labels 的 log_probs
log_probs = [-1.5,  # log P(token_1=1 | context)
             -2.0,  # log P(token_2=2 | context)
             -1.0,  # log P(token_3=2 | context)
             -1.1]  # log P(token_4=0 | context)
```

### 6.8 Pad 回原始 shape（第 505-522 行）

```python
if calculate_entropy:
    full_entropy = pad_input(
        hidden_states=entropy_rmpad.unsqueeze(-1),
        indices=indices,  # 来自 unpad_input
        batch=batch_size,
        seqlen=seqlen,
    )

full_log_probs = pad_input(
    hidden_states=log_probs.unsqueeze(-1),
    indices=indices,
    batch=batch_size,
    seqlen=seqlen,
)

# 只返回 response 部分
if calculate_entropy:
    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
```

**逐行解析**:

**pad_input() 函数**:
- **作用**: 将 packed 格式还原回 (batch_size, seqlen) 格式
- **原理**: 使用 `indices` 将 tokens 放回原位置，其他位置填充 0

**第 521-522 行**: `[:, -response_length - 1 : -1]`
- **作用**: 只保留响应部分的 log_probs
- **为什么 -1？** 因为最后一个 token 没有下一个 token 可预测
- **示例**:
  ```python
  total_length = 10 (prompt=6, response=4)
  response_length = 4

  input_ids:    [p1, p2, p3, p4, p5, p6, r1, r2, r3, r4]
  log_probs:    [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]
                                        ^           ^
                                        提取 l7-l9

  [:, -5:-1] = [:, -response_length-1:-1] = [l7, l8, l9]
  ```

### 6.9 返回结果

```python
return log_probs, entropys
```

**返回值**:
- `log_probs`: shape=(batch_size, response_length)
- `entropys`: shape=(batch_size, response_length) 或 None

---

## 7. 自定义模型实现

### 7.1 为什么需要自定义模型？

标准 Transformers 模型：
```python
output = model(input_ids, attention_mask)
# output.logits: 只有最后一层的输出
# output.hidden_states: 需要手动设置 output_hidden_states=True
```

**问题**:
1. 无法直接获取中间层的 logits
2. 需要手动计算 `H^k E^T_u`

BuPO 自定义模型：
```python
output = custom_model(input_ids, attention_mask, output_hidden_states=True)
# output.logits: 最后一层
# output.hidden_states: 所有层的隐藏状态 (自动)
# output.mid_layer_logits: 内部层的 logits (自动计算)
```

### 7.2 自定义模型位置

```
verl/models/custom_model/
├── modeling_qwen2.py
├── modeling_qwen3.py      # ← 我们重点看这个
├── modeling_llama.py
├── configuration_qwen2.py
├── configuration_qwen3.py
└── configuration_llama.py
```

### 7.3 Qwen3ForCausalLM.forward() 方法

```
文件: verl/models/custom_model/modeling_qwen3.py
方法: Qwen3ForCausalLM.forward()
位置: 第 523-597 行
```

#### 核心实现（第 585-597 行）

```python
# 第 523-584 行: 标准的 forward 逻辑（与 HuggingFace 相同）
outputs = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    ...
    output_hidden_states=output_hidden_states,  # 输出所有层
    ...
)

hidden_states = outputs[0]  # 最后一层的隐藏状态
logits = self.lm_head(hidden_states)  # 最后一层的 logits

# 第 577-584 行: 创建标准输出对象
output = CausalLMOutputWithPastNew(
    loss=loss,
    logits=logits,
    past_key_values=outputs.past_key_values,
    hidden_states=outputs.hidden_states,  # 所有层的隐藏状态
    attentions=outputs.attentions,
)

# ============ BuPO 添加的核心代码 ============
internal_logits = {}  # 第 585 行

"""
Extraction of Internal Hidden States

args:
    startk: which layer used as internal layer policy.
            For Qwen3-4B, startk ∈ [0, 35].
            Here, startk = 0 equals to layer 1 in paper.
"""
startk = int(self.config.internal_layer)  # 第 593 行
# 例如: startk = 5

for i in range(startk, startk+1):  # 第 594 行
    # 只循环一次，计算第 startk 层的 logits
    internal_logits[i] = self.lm_head(outputs.hidden_states[i+1][:, slice_indices, :])  # 第 595 行
    # ↑ 这就是论文中的 H^k E^T_u (公式 6)

output.mid_layer_logits = internal_logits  # 第 596 行
return output  # 第 597 行
```

**逐行详细解析**:

**第 585 行**: `internal_logits = {}`
- 创建空字典，用于存储内部层的 logits

**第 593 行**: `startk = int(self.config.internal_layer)`
- **从配置读取**: 要计算哪一层的内部 logits
- **配置路径**: `actor_rollout_ref.model.override_config.internal_layer=5`
- **示例**: startk = 5 表示第 5 层（代码从 0 开始）

**第 594 行**: `for i in range(startk, startk+1):`
- **range(5, 6)**: 只循环一次，i=5
- **为什么不直接写 i=startk？**
  - 代码设计为可以计算多层，但当前只用一层

**第 595 行**: `internal_logits[i] = self.lm_head(outputs.hidden_states[i+1][:, slice_indices, :])`
- **这是 BuPO 最核心的一行代码！**
- 让我们分解：

**1. `outputs.hidden_states`**:
- **类型**: Tuple of Tensors
- **长度**: L + 1（L 是 Transformer 层数）
- **内容**:
  ```python
  hidden_states[0]:  Embedding 层输出
  hidden_states[1]:  Layer 0 的输出
  hidden_states[2]:  Layer 1 的输出
  ...
  hidden_states[k+1]: Layer k 的输出  # ← 我们要这个
  ...
  hidden_states[L]:  Layer L-1 的输出（最后一层）
  ```

**2. `outputs.hidden_states[i+1]`**:
- **i=5**: `hidden_states[6]` = Layer 5 的输出
- **为什么 +1？** 因为 `hidden_states[0]` 是 embedding

**3. `[:, slice_indices, :]`**:
- **作用**: 只提取需要计算 logits 的部分
- **slice_indices**: 通常是响应部分的索引

**4. `self.lm_head(...)`**:
- **lm_head**: Language Model Head（输出投影层）
- **作用**: 将隐藏状态投影到词表空间
- **数学**: `logits = H^k W^T` （其中 W 是 lm_head 的权重，也就是 E_u）
- **输入 shape**: (batch_size, seq_len, hidden_dim)
- **输出 shape**: (batch_size, seq_len, vocab_size)

**完整流程图**:
```
Layer 5 的输出 (H^5)
    ↓ shape: (batch, seq, hidden_dim)
self.lm_head (E_u^T)
    ↓ 矩阵乘法: H^5 @ E_u^T
logits^5
    ↓ shape: (batch, seq, vocab_size)
softmax(logits^5) → π^5_Layer (内部层策略)
```

**这对应论文的公式 6**:
```
π^l_Layer ≡ P^l_Layer = softmax(H^l E^T_u)
```

**第 596 行**: `output.mid_layer_logits = internal_logits`
- 将计算的内部 logits 添加到输出对象
- **mid_layer_logits[5]**: 第 5 层的 logits

### 7.4 CausalLMOutputWithPastNew 类

```
文件: verl/models/custom_model/modeling_qwen3.py
位置: 第 88-120 行
```

```python
class CausalLMOutputWithPastNew(ModelOutput):
    """
    扩展的输出类，添加了 mid_layer_logits 字段
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[UserDict[str, Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mid_layer_logits: Optional[UserDict[int, torch.FloatTensor]] = None  # ← 新增
```

**mid_layer_logits**:
- **类型**: 字典 `{layer_index: logits_tensor}`
- **示例**:
  ```python
  mid_layer_logits = {
      5: tensor of shape (batch, seq, vocab_size)
  }
  ```

### 7.5 模型替换机制

```
文件: verl/workers/fsdp_workers.py
位置: 第 249-256 行
```

```python
if hasattr(self.config.actor, "internal_policy_interative") and \
   self.config.actor.internal_policy_interative:
    # 导入自定义模型
    from verl.models.custom_model import modeling_qwen2 as custom_modeling_qwen2
    from verl.models.custom_model import modeling_qwen3 as custom_modeling_qwen3
    from verl.models.custom_model import modeling_llama as custom_modeling_llama

    # 替换 sys.modules 中的模型
    sys.modules["transformers.models.qwen2.modeling_qwen2"] = custom_modeling_qwen2
    sys.modules["transformers.models.qwen3.modeling_qwen3"] = custom_modeling_qwen3
    sys.modules["transformers.models.llama.modeling_llama"] = custom_modeling_llama
```

**逐行解析**:

**第 249 行**: 检查是否启用 BuPO

**第 250-252 行**: 导入自定义模型模块

**第 253-255 行**: **关键的模块替换**
- **sys.modules**: Python 的模块缓存
- **作用**: 将 HuggingFace 的标准模型替换为自定义模型
- **原理**: 当后续代码 `import transformers.models.qwen3.modeling_qwen3` 时，实际导入的是自定义版本

**为什么这样做？**
- HuggingFace 的 AutoModel 会自动导入模型
- 通过替换 sys.modules，无需修改 HuggingFace 的代码
- 保持兼容性，切换方便

---

## 8. Loss 计算与反向传播

### 8.1 PPO Loss 计算（第 925-933 行）

```python
# 获取 policy loss 计算函数
loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
policy_loss_fn = get_policy_loss_fn(loss_mode)

# 计算 policy loss
pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
    old_log_prob=old_log_prob,  # 来自 rollout 的 old policy log prob
    log_prob=log_prob,  # 当前 policy 的 log prob
    advantages=advantages,  # 优势函数 (GRPO 计算)
    response_mask=response_mask,  # 响应部分的 mask
    loss_agg_mode=loss_agg_mode,  # "token-mean" 或 "sample-mean"
    config=self.config,
    rollout_log_probs=rollout_log_probs,
)
```

**逐行解析**:

**policy_loss_fn() 函数** (vanilla PPO loss):
```python
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, ...):
    """
    论文公式 4 (GRPO) 的实现
    """
    # 1. 计算 importance ratio
    ratio = torch.exp(log_prob - old_log_prob)  # r_i,t = π_θ / π_θ_old

    # 2. 计算 clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - ε, 1 + ε)  # clip(r, 1-ε, 1+ε)

    # 3. 计算 surrogate loss
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    pg_loss = -torch.min(loss1, loss2)  # min{r*A, clip(r)*A}

    # 4. 聚合 loss
    pg_loss = agg_loss(pg_loss, response_mask, loss_agg_mode)

    return pg_loss, ...
```

**关键点**:

**Phase 1 (BuPO) 的 importance ratio**:
```python
# log_prob 来自 _forward_micro_batch_layer_k()
r̂_i,t = exp(log_prob_k - old_log_prob_k)
      = π^k_Layer(o_t | context) / π^k_Layer,old(o_t | context)
```

**Phase 2 的 importance ratio**:
```python
# log_prob 来自 _forward_micro_batch()
r_i,t = exp(log_prob - old_log_prob)
      = π_θ(o_t | context) / π_θ_old(o_t | context)
```

**这实现了论文公式 10（InterGRPO）的 importance ratio 切换！**

### 8.2 添加 Entropy Regularization（第 935-940 行）

```python
if entropy_coeff != 0:
    entropy_loss = agg_loss(
        loss_mat=entropy,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode
    )
    # 总 loss = policy loss - entropy_coeff * entropy_loss
    policy_loss = pg_loss - entropy_loss * entropy_coeff
else:
    policy_loss = pg_loss
```

**作用**:
- **Entropy Regularization**: 鼓励策略保持探索性
- **entropy_coeff**: 熵的权重（通常设为 0，BuPO 不使用）

### 8.3 反向传播（第 951-965 行）

```python
# 缩放 loss（梯度累积）
loss = policy_loss * loss_scale_factor
loss.backward()  # 反向传播

# 梯度裁剪
if self.config.grad_clip > 0:
    if self.config.strategy == "fsdp2":
        fsdp2_clip_grad_norm_(
            self.actor_module,
            max_norm=self.config.grad_clip
        )
    else:
        torch.nn.utils.clip_grad_norm_(
            self.actor_module.parameters(),
            self.config.grad_clip
        )

# 更新参数
self.actor_optimizer.step()
self.actor_scheduler.step()
```

**逐行解析**:

**loss.backward()**:
- **反向传播**: 计算梯度
- **关键**:
  - **Phase 1**: 只有 layers 0-k 的参数会收到梯度
  - **Phase 2**: 所有层的参数都会收到梯度

**为什么 Phase 1 只更新 0-k 层？**

**原因**: 残差连接（Residual Connection）的梯度流

```python
# 第 k 层的输出
H^k = H^0 + A^1 + F^1 + ... + A^k + F^k

# 对第 k+1 层的参数 θ_{k+1} 求导
∂H^k / ∂θ_{k+1} = 0  # 因为 H^k 不依赖 θ_{k+1}

# 因此第 k+1 层及以上的参数梯度为 0，不会被更新
```

**这实现了论文公式 16 的梯度流控制！**

**梯度裁剪**:
- **作用**: 防止梯度爆炸
- **max_norm**: 梯度的最大范数（默认 1.0）

**optimizer.step()**:
- **AdamW 优化器**: 根据梯度更新参数

---

## 9. 完整训练循环

### 9.1 单个训练 Step 的流程

```
RayPPOTrainer.train_step(step)
    ↓
1. Rollout Phase (生成响应)
    └─ rollout_module.generate()  # vLLM 生成
    └─ 返回: responses, rollout_log_probs
    ↓
2. Reward Computation
    └─ reward_manager.compute_rewards(responses)
    └─ 返回: rewards
    ↓
3. Compute Advantages (GRPO)
    └─ advantages = (rewards - mean(rewards)) / std(rewards)
    ↓
4. Actor Update (BuPO 核心)
    └─ actor.update_policy(data)
        ├─ 判断 global_steps <= iterative_steps?
        ├─ Phase 1: _forward_micro_batch_layer_k()
        │   └─ 计算 π^k_Layer 的 log_probs
        │   └─ 计算 PPO loss
        │   └─ 反向传播（只更新 0-k 层）
        └─ Phase 2: _forward_micro_batch()
            └─ 计算 π_θ 的 log_probs
            └─ 计算 PPO loss
            └─ 反向传播（更新所有层）
    ↓
5. Update Reference Policy (可选)
    └─ ref_policy.sync_from_actor()
    ↓
6. Log Metrics & Save Checkpoint
```

### 9.2 meta_info['global_steps'] 的维护

**问题**: `global_steps` 是如何传递和更新的？

**答案**: 在 RayPPOTrainer 中维护

```python
# verl/trainer/ppo/ray_trainer.py (大致)
class RayPPOTrainer:
    def __init__(self, ...):
        self.global_steps = 0

    def train(self):
        for step in range(total_training_steps):
            self.global_steps += 1

            # 生成 rollout data
            data = self.rollout(...)

            # 添加 global_steps 到 meta_info
            data.meta_info['global_steps'] = self.global_steps

            # Actor update
            self.actor.update_policy(data)
```

**传递路径**:
```
RayPPOTrainer.global_steps
    ↓ (添加到 data.meta_info)
DataProto.meta_info['global_steps']
    ↓ (split 后保留)
micro_batch.meta_info['global_steps']
    ↓ (在 dp_actor.py 中判断)
if micro_batch.meta_info['global_steps'] <= iterative_steps:
```

---

## 10. 配置参数详解

### 10.1 BuPO 相关配置

```yaml
# verl/trainer/config/actor/dp_actor.yaml
actor:
  internal_policy_interative: False  # 启用 BuPO
  internal_layer: 5                   # 优化第 5 层
  iterative_steps: 30                 # Phase 1 持续 30 步
```

### 10.2 在训练脚本中的设置

```bash
# run_code/BuPO_qwen3.sh
k=5                     # 内部层索引
iterative_steps=30      # Phase 1 步数

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.internal_policy_interative=True \
    actor_rollout_ref.actor.internal_layer=${k} \
    actor_rollout_ref.actor.iterative_steps=${iterative_steps} \
    actor_rollout_ref.model.override_config.internal_layer=${k}
```

**参数解释**:

| 参数 | 含义 | 示例值 | 说明 |
|------|------|--------|------|
| `internal_policy_interative` | 启用 BuPO | True | False 表示标准 GRPO |
| `internal_layer` | 优化的层索引 | 5 | Qwen3-4B 有 36 层 (0-35) |
| `iterative_steps` | Phase 1 步数 | 30 | 前 30 步优化内部层 |
| `model.override_config.internal_layer` | 模型配置 | 5 | 告诉模型计算第 5 层 logits |

### 10.3 如何选择 internal_layer？

**论文建议**:

1. **Qwen 系列**:
   - Qwen3-4B (36 层): layer 5 或 6
   - Qwen3-8B (36 层): layer 5 或 6
   - **原则**: 选择 FFN entropy change 开始为 0 的层（Integration 阶段的开始）

2. **Llama 系列**:
   - Llama-3B (28 层): layer 27（倒数第二层）
   - Llama-8B (32 层): layer 31（倒数第二层）
   - **原则**: Llama 的 FFN entropy 一直为正，选择最后有正 entropy 的层

**查看 entropy 动态**:
```bash
python visualization/plot_internal_entropy.py
# 查看 Figure 3 (Entropy Change)，选择 ΔH^l_FFN ≈ 0 的层
```

---

## 11. 常见问题 FAQ

### Q1: BuPO 为什么有效？

**答案**:
1. **底层特征优先对齐**: Phase 1 优化底层，让底层学会高层次推理
2. **渐进式训练**: 先对齐底层特征，再对齐整体策略
3. **更稳定的训练**: 底层提供更好的特征基础

### Q2: 为什么需要两次切换逻辑（806 行和 906 行）？

**答案**:
- **第 806 行** (`compute_ref_log_prob`): 计算参考 log_probs（无梯度）
- **第 906 行** (`update_policy`): 策略更新（有梯度）
- 两者都需要判断当前阶段，保持一致

### Q3: internal_layer 设为 -1 会怎样？

**答案**:
- 会导致错误，因为 `hidden_states[-1+1]` = `hidden_states[0]` = embedding
- 应该设置为有效的层索引 (0 到 L-1)

### Q4: 能同时优化多个内部层吗？

**答案**:
- 代码支持（第 594 行的 for 循环）
- 但论文只优化一层
- 多层优化可能导致训练不稳定

### Q5: Phase 2 还能切回 Phase 1 吗？

**答案**:
- 不能，这是单向的
- `global_steps` 单调递增，一旦超过 `iterative_steps` 就不会回退

### Q6: 梯度裁剪的 max_norm=1.0 是怎么确定的？

**答案**:
- 经验值，PPO 训练通常使用 1.0
- 可以调整，但不建议超过 5.0

---

## 12. 调试技巧

### 12.1 打印 global_steps

```python
# 在 dp_actor.py 的第 807 行添加
print(f"[DEBUG] global_steps={micro_batch.meta_info['global_steps']}, "
      f"iterative_steps={self.config.iterative_steps}, "
      f"Phase={'1' if micro_batch.meta_info['global_steps'] <= self.config.iterative_steps else '2'}")
```

### 12.2 验证内部层 logits

```python
# 在 modeling_qwen3.py 的第 596 行添加
print(f"[DEBUG] internal_layer={startk}, "
      f"mid_layer_logits shape={internal_logits[startk].shape}")
```

### 12.3 检查梯度流

```python
# 在 dp_actor.py 的第 951 行后添加
for name, param in self.actor_module.named_parameters():
    if param.grad is not None:
        print(f"[DEBUG] {name}: grad_norm={param.grad.norm().item():.4f}")
```

---

## 13. 总结

### 13.1 BuPO 核心要点

1. **Two-Phase Training**:
   - Phase 1: 优化 π^k_Layer (internal layer policy)
   - Phase 2: 优化 πθ (full model policy)

2. **关键实现**:
   - 自定义模型: 计算 `mid_layer_logits`
   - 两阶段切换: 判断 `global_steps <= iterative_steps`
   - 梯度流控制: 残差连接自动限制梯度范围

3. **代码位置**:
   - 训练入口: `verl/trainer/main_ppo.py`
   - Actor 实现: `verl/workers/actor/dp_actor.py:806-820, 906-920`
   - 内部层前向: `verl/workers/actor/dp_actor.py:357-555`
   - 自定义模型: `verl/models/custom_model/modeling_qwen3.py:585-596`

### 13.2 学习路径建议

1. **理解标准 PPO/GRPO**: 先熟悉基础 RL 算法
2. **阅读论文 Section 3-5**: 理解 Internal Policy 概念
3. **运行可视化**: 查看 entropy 动态
4. **单步调试**: 在关键位置添加 print，观察数据流
5. **修改参数**: 尝试不同的 `internal_layer` 和 `iterative_steps`

### 13.3 扩展阅读

- **verl 框架文档**: https://github.com/volcengine/verl
- **PPO 论文**: https://arxiv.org/abs/1707.06347
- **GRPO 论文**: https://arxiv.org/abs/2402.03300
- **Flash Attention**: https://arxiv.org/abs/2205.14135

---

**文档版本**: 1.0
**作者**: Claude (Anthropic)
**最后更新**: 2026-01-06
**对应代码版本**: BuPO v1.0
**字数统计**: ~18,000 字
