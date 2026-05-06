# INS Mechanization (惯性导航机械编排) 详解

## 一、是什么：一句话定义

> **INS Mechanization** 就是用 IMU 的角速度/加速度测量值，**一步接一步地递推**出载体的**姿态、速度和位置**。
>
> 它是惯性导航系统的"预测步骤"——相当于蒙着眼睛走路的数学模型。

---

## 二、函数签名

```cpp
void MISC::insMechanization(
    const IntegrationConfiguration &config,  // 配置（重力、地球模型等）
    const IMU &imu_pre,                      // 上一时刻的IMU测量
    const IMU &imu_cur,                      // 当前时刻的IMU测量
    IntegrationState &state)                 // [in/out] 输入旧状态，输出新状态
```

**输入输出关系：**

```
旧状态(p, v, q, bg, ba) + [IMU_pre, IMU_cur]  →  新状态(p', v', q', bg, ba)
```

---

## 三、IntegrationState 数据结构

来自 `common/integration_state.h`：

| 字段 | 含义 | 单位 |
|------|------|------|
| `p` | 位置 (3D) | m |
| `q` | 姿态四元数 (3D) | - |
| `v` | 速度 (3D) | m/s |
| `bg` | 陀螺零偏 (3D) | rad/s |
| `ba` | 加速度计零偏 (3D) | m/s² |
| `s` | 比例因子 (3D) | - |
| `sodo` | 里程计比例因子 | - |
| `avb` | 安装角 (2D) | rad |
| `sg` | 陀螺比例因子 (3D) | - |
| `sa` | 加速度计比例因子 (3D) | - |

---

## 四、代码逐段精讲

源码位置：`ff_lins/common/misc.cc:138-184`

### 第①步：零偏补偿（第142-146行）

```cpp
imu_cur2.dtheta = imu_cur.dtheta - imu_cur.dt * state.bg;
imu_cur2.dvel   = imu_cur.dvel   - imu_cur.dt * state.ba;
imu_pre2.dtheta = imu_pre.dtheta - imu_pre.dt * state.bg;
imu_pre2.dvel   = imu_pre.dvel   - imu_pre.dt * state.ba;
```

- IMU 原始数据包含常值零偏误差
- 在积分之前用当前估计的零偏 `bg`、`ba` 扣除
- 当前/前一刻的 IMU 都需要做零偏补偿（双子样要用到两个历元）

### 第②步：双子样圆锥/划船补偿（第152-154行）

```cpp
// 速度增量：划船效应补偿
Vector3d dvfb = imu_cur2.dvel
    + 0.5 * imu_cur2.dtheta.cross(imu_cur2.dvel)
    + 1.0/12.0 * (imu_pre2.dtheta.cross(imu_cur2.dvel)
                + imu_pre2.dvel.cross(imu_cur2.dtheta));

// 角度增量：圆锥效应补偿
Vector3d dtheta = imu_cur2.dtheta
    + 1.0/12.0 * imu_pre2.dtheta.cross(imu_cur2.dtheta);
```

- **为什么要双子样？** 捷联惯导中，载体角振动会导致圆锥效应误差，线振动会导致划船效应误差
- 只用单子样（梯形积分）在高动态下精度不够
- 双子样利用相邻两个 IMU 历元的叉乘项修正，将误差降低到 O(Δt⁴) 量级
- 这是捷联惯导算法的标准做法

### 第③步：姿态更新（第170-178行）

**无地球模型模式（默认，`config.iswithearth = false`）：**

```cpp
state.q *= Rotation::rotvec2quaternion(dtheta);
state.q.normalize();
```

- 补偿后的角增量 `dtheta` 是旋转向量
- 旋转向量 → 四元数，左乘到当前姿态上
- 归一化防止四元数漂移

**有地球模型模式：**

```cpp
Vector3d dnn    = -config.iewn * dt;
Quaterniond qnn = Rotation::rotvec2quaternion(dnn);
state.q = qnn * state.q * Rotation::rotvec2quaternion(dtheta);
```

- 额外考虑地球自转（`iewn` 是地球自转角速度在导航系下的投影）
- 先补偿地球自转，再做姿态更新

### 第④步：速度更新（第167-178行）

**无地球模型模式：**

```cpp
dvel = state.q.toRotationMatrix() * dvfb + config.gravity * dt;
//     ↑ 比力转到导航系(n系)          ↑ 重力加速度积分
state.v += dvel;
```

- `dvfb` 是载体坐标系下的比力增量
- 用当前姿态旋转矩阵转到导航坐标系
- 加上重力项得到导航系下的总速度增量

**有地球模型模式：**

```cpp
Vector3d dv_cor_g = (config.gravity - 2.0 * config.iewn.cross(state.v)) * dt;
dvel = 0.5 * (I + qnn.toRotationMatrix()) * state.q.toRotationMatrix() * dvfb + dv_cor_g;
```

- 额外计入哥氏加速度补偿（-2ω×v）
- 地球自转补偿项 `qnn` 引起的比力方向变化也考虑了

### 第⑤步：位置更新（第181-183行）

```cpp
state.p += dt * state.v + 0.5 * dt * dvel;   // 梯形积分
state.v += dvel;                               // 最后更新速度
```

- 位置用**前后历元平均速度**做梯形积分
- 顺序是先算位置增量（用旧速度+本次速度增量的一半），再更新速度

---

## 五、三函数调用链

```
LINS::processLidarFrame()              ← 处理每一帧LiDAR
  └─ LINS::waitImuDoInsMechanization() ← 确保IMU推进到指定时间
       └─ LINS::doInsMechanization()   ← 从IMU缓冲队列循环取数据
            └─ MISC::insMechanization()← ◀ 核心单步递推

优化器更新状态后（重递推）：
  └─ MISC::redoInsMechanization()      ← 从优化后的状态点重算
       └─ MISC::insMechanization()     ← ◀ 也是它
```

### doInsMechanization
源码：`ff_lins/ff_lins/ff_lins.cc:430-463`

- 从 IMU 缓冲队列 (`imu_buffer_`) 一次性取到目标时间
- 对每个 IMU 依次调用 `insMechanization` 递推状态
- 将结果推入 `ins_window_` 供 LiDAR 帧使用
- 异步写入轨迹文件

### redoInsMechanization
源码：`ff_lins/common/misc.cc:186-236`

- 优化器修正了某个时间点的状态后调用
- 找到该时间点在 `ins_window_` 中的位置
- 从那个位置开始重新往前做机械编排
- 清理过期的 IMU 历元

---

## 六、直观类比

| 组件 | 比喻 |
|------|------|
| IMU（陀螺+加速度计） | 蒙着眼，手里拿着陀螺仪和加速度计 |
| 零偏补偿 | 知道仪器有系统误差，先校准 |
| 姿态更新 | 感觉身体转了某个角度和方向 |
| 比力积分 | 感觉身体在某个方向加速 |
| 速度更新 | 算出自己在朝哪个方向以多快速度运动 |
| 位置更新 | 估算自己已经走了多远 |
| LiDAR 帧到来 | 有人告诉你"你现在在 xx 位置" |
| 图优化 | 根据 LiDAR 观测修正当前状态估计 |
| 重递推 (redo) | 修正后，从修正点之间重新往前算一遍 |

---

## 七、在 FF-LINS 系统中的地位

```
┌─────────────────────────────────────────────────────┐
│                    FF-LINS 系统                       │
│                                                       │
│   IMU 输入 ──→  insMechanization ──→ 预测状态         │
│                     (预测)           ↓                │
│                                      ↓                │
│   LiDAR 输入 ──→   图优化/因子图   ──→ 校正状态       │
│                     (校正)           ↓                │
│                                      ↓                │
│                     redoInsMechanization              │
│                      (重递推，保持一致性)              │
└─────────────────────────────────────────────────────┘
```

- **insMechanization** = 系统的**时间更新（预测）**
- **LiDAR因子 + 优化器** = 系统的**观测更新（校正）**
- **redoInsMechanization** = 保证校正后的**帧间一致性**
