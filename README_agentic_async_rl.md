# MiniMind Agentic Async RL

<div align="center">

[English](./README_agentic_async_rl_en.md) | 中文

</div>


---

## 1. 这是什么

`train_agentic_rl_async.py` 是 MiniMind 在多轮 Tool-Use 场景下的 **异步 Agentic RL** 训练脚本。

它继承了 `train_agent.py` 里已经实现好的三件事：

- 多轮工具调用轨迹 rollout
- 基于 `gt` / 格式 / 工具执行结果的延迟奖励
- rollout engine 与 learner 的解耦

但进一步把训练范式从：

```text
先采样一整批轨迹 -> 再统一更新一次策略
```

推进到了：

```text
后台持续生产 rollout -> 前台持续消费 fresh trajectories -> 按策略版本做近似 on-policy 更新
```

这使它更接近真实的大规模 Agent RL 系统，而不只是“支持多轮工具调用的同步 PPO/GRPO”。

---

## 2. 它和 `train_agent.py` 的核心差异

### 2.1 同步版：`train_agent.py`

`train_agent.py` 的主循环更接近经典 online RL：

```text
rollout batch -> reward -> update policy -> sync rollout policy -> next batch
```

特点是：

- 每次更新前，都会先等一整个 batch rollout 完成
- rollout 使用的基本是同一版策略
- 逻辑直观，容易调试
- 但训练端经常要等待采样端，吞吐容易卡在 rollout

### 2.2 异步版：`train_agentic_rl_async.py`

异步版引入了一个后台 `AsyncRolloutManager`：

- 多个 producer 线程持续从数据池中取样本
- 后台不断展开多轮轨迹并写入队列
- learner 端从队列中拿到“足够新鲜”的 group 直接更新
- 训练与 rollout 不再严格 step-by-step 锁步

直接收益是：

- 更少的训练空转等待
- 更高的 rollout 利用率
- 更自然地支持 trainer / rollout 分工
- 更接近生产级 agent training 的执行方式

---

## 3. 异步 Agentic RL 真正多出来了什么

很多“异步 RL”只是在同步循环外面套了线程池，这个脚本不是。它额外补齐了异步训练真正需要的几层机制。

### 3.1 策略版本跟踪

每条轨迹、每个 token 都会记录它来自哪个 `policy_version`：

- `PolicyVersion`
- `response_versions`
- `per_token_versions`
- `group.policy_version`

这意味着 learner 在更新时知道：

- 当前 loss 正在消费哪一版行为策略生成的数据
- 这些数据离当前模型已经“陈旧”了多少步

这是一切异步 PPO 近似成立的基础。

### 3.2 陈旧样本控制，而不是盲目吃队列

异步最大的问题不是“线程安全”，而是 **off-policy 污染**。

这个脚本用三层方式控制它：

- `max_version_gap`：超过允许版本差的 group 直接丢弃
- `resolve_prox_logps(...)`：对旧策略 logprob 做近似修正
- `async_stale_discount`：越旧的 trajectory，在 loss 里的权重越低

也就是说，它不是把旧样本全扔掉，也不是全量相信旧样本，而是在“利用吞吐”和“约束偏移”之间找折中。

### 3.3 trainer-rollout 分工

脚本支持：

- `multi_gpu_mode=ddp`
- `multi_gpu_mode=trainer_rollout`

其中 `trainer_rollout` 是更像工业实现的模式：

- 1 张 rank 主要负责 learner update
- 其余 rank 更偏向 rollout producer
- 中间通过队列、prefetch、group fetch 做衔接

这比单纯 DDP 下每卡都自己 rollout 再自己更新，更适合 Tool-Use 这种 rollout 侧更重的任务。

### 3.4 真正围绕异步暴露监控指标

同步脚本主要看 reward / kl / entropy。

这个异步脚本还会显式记录：

- `async/staleness_group_mean`
- `async/staleness_group_max`
- `async/token_staleness_*`
- `async/row_weight_mean`
- `async/queue_utilization`
- `async/avg_rollout_sec`
- `async/avg_train_wait_sec`
- `async/train_wait_sec_p90`
- `async/online_group_accept_rate`
- `async/stale_drop_rate`

这很重要，因为异步训练是否跑得好，不能只看 reward，还要看：

- 队列是不是一直空
- learner 是否长期等 rollout
- stale 样本是不是越堆越多
- 采样吞吐和训练吞吐是否匹配





## 4. 训练流程概览

```text
Prompt Pool
   -> AsyncRolloutManager
   -> rollout workers / rollout ranks
   -> trajectory queue
   -> pack_groups
   -> compute_async_loss
   -> optimizer step
   -> policy sync
   -> next fresh trajectories
```

如果只看最关键的数据流，可以理解为：

```text
sample prompt
-> multi-turn tool rollout
-> delayed reward on full trajectory
-> stale-aware PPO-style update
-> periodic rollout policy sync
```



## 5. 实验结果怎么看

### 5.1 在线实验面板

公开实验记录：

- [SwanLab: MiniMind-Agentic-Async-RL](https://swanlab.cn/@xiaoxuan/MiniMind-Agentic-Async-RL?utm_source=website_qr&utm_medium=qr_scan)

建议在面板里重点看这几组曲线：

- `reward/mean` 与 `eval/reward`
- `train/accuracy_exact_match`、`train/accuracy_answer_cov`
- `train/tool_success_rate`、`train/tool_correct_choice_rate`
- `train/tool_value_order_cov`
- `policy/kl_ref_k3`、`policy/ratio_clip_frac`
- `async/avg_rollout_sec`、`async/avg_train_wait_sec`
- `async/queue_utilization`
- `async/staleness_group_mean`

### 5.2 结果应该如何解读

如果异步训练是健康的，通常会看到：

- reward 持续上升，而不是很早进入震荡
- exact match / answer coverage 随训练同步改善
- tool success 与 tool value coverage 一起上升
- queue utilization 不是长期贴近 0
- train wait 不会持续高于 rollout 本身
- staleness 受控，不会无上限飙升

也就是说，异步优化的目标不是单纯“更快”，而是：

- 在保持策略更新稳定的前提下
- 尽量减少 learner 等待 rollout 的空窗

### 5.3 本地可确认的运行信息

仓库内 `swanlog` 记录显示，该脚本至少存在如下公开运行痕迹：

- 命令：`trainer/train_agentic_rl_async.py --rollout_engine torch`
- 环境：`Python 3.10.20`
- GPU：`4 x Tesla V100S-PCIE-32GB`
- 可视化：`swanlab 0.7.16`

这说明它不是停留在设计草图阶段，而是被实际跑过并接入了完整日志系统。

---

## 6. 默认训练配置的风格

从源码默认参数看，这个脚本的目标非常明确：先把 **小模型数学工具调用 agent** 训稳定。

默认配置包括：

- 数据：`dataset/agent_rl_math.jsonl`
- 工具子集：`calculate_math,unit_converter,get_current_time`
- `num_generations=4`
- `max_turns=3`
- `max_gen_len=192`
- `use_decoupled_ppo=1`
- `beta_auto_tune=1`
- `bc_coef>0`，带有辅助 BC 稳定项

这套默认值不是为了做通用 Agent benchmark，而是为了让 MiniMind 这种小模型在可验证 Tool-Use 任务上先学会：

- 该不该调用工具
- 调哪个工具
- 调完之后如何把结果组织进最终答案

---

## 7. 快速开始

### 7.1 Torch rollout

```bash
torchrun --nproc_per_node=4 trainer/train_agentic_rl_async.py --rollout_engine torch
```

或

```bash
python trainer/train_agentic_rl_async.py --rollout_engine torch
```

### 7.2 SGLang rollout

先启动服务：

```bash
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
```

再训练：

```bash
python trainer/train_agentic_rl_async.py \
  --rollout_engine sglang \
  --sglang_base_url http://localhost:8998 \
  --sglang_shared_path ./sglang_ckpt_agentic_async
```

### 7.3 如果你只想先理解思路

建议按这个顺序看源码：

1. `trainer/train_agent.py`
2. `trainer/train_agent_async.py`
3. `trainer/train_agentic_rl_async.py`

这样会更容易看出它是如何从：

- 同步 Agentic RL
- 到单机异步队列
- 再到带策略版本控制的 trainer-rollout 异步系统

一步步演化过来的。

---


