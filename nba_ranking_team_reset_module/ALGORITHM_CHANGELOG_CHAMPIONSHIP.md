# 冠军含金量与奖项权重算法重置说明

本版本解决上一版“累计数据与高级数据过强、无冠球员容易被排过高”的问题，重点改动如下。

## 1. 奖项层级重置

奖项不再统一折算，而是按含金量分层：

```text
MVP >= FMVP >= All-NBA 1st >= DPOY > All-NBA 2nd > All-NBA 3rd / All-Defense > All-Star
```

单赛季奖项分：

```text
MVP Share Score  = 35 * MVP Share
FMVP Share Score = 30 * FMVP Share
DPOY Share Score = 20 * DPOY Share
All-NBA 1st      = 24
All-NBA 2nd      = 13
All-NBA 3rd      = 7
All-Defense 1st  = 6
All-Defense 2nd  = 3.5
All-Star         = 2
```

## 2. 冠军分层：单核 / 多核 / 核心 / 轮换

冠军不再统一加分，而是识别角色：

### 单核冠军定义

```text
球队夺冠
且球员本人为 All-NBA 球员
且同队没有其他 All-NBA 球员
```

### 冠军角色识别

```text
冠军老大：队内 season_ability_value / playoff_ws / playoff_mp 至少一项排名第 1
冠军核心：队内 season_ability_value / playoff_ws / playoff_mp 至少一项排名前 3
冠军轮换：夺冠队中非核心但赛季出场时间足够的球员
```

### 单赛季冠军得分

```text
单核冠军老大：34
单核冠军核心/重要队友：13
单核冠军普通成员：5
多核冠军老大：25
多核冠军核心：18
多核冠军其他成员：4.5
```

## 3. 历史地位分重置

历史地位分更重视冠军和奖项：

```text
Legacy Score Rule =
0.29 * Championship Score
+ 0.25 * Award Score
+ 0.10 * Legacy Peak 3 Score
+ 0.08 * Career Per Game Score
+ 0.07 * Stage Weighted Legacy Value
+ 0.05 * Playoff Score
+ 0.05 * Longevity Score
+ 0.05 * Media Reputation Score
+ 0.03 * Environment Difficulty Score
+ 0.35 * Career Stage Bonus
+ HOF Bonus
```

注意：非冠军季后赛轮次仍然有价值，但已经被打折，低于冠军本身。

## 4. 纯实力分重置

纯实力也考虑冠军核心证明，但仍保留巅峰表现、场均能力和攻防能力：

```text
Ability Score Rule =
0.17 * Peak 1
+ 0.18 * Peak 3
+ 0.13 * Prime Score
+ 0.08 * Peak 5
+ 0.10 * Career Per Game Score
+ 0.13 * Championship Score
+ 0.06 * Playoff Score
+ 0.06 * Offense Score
+ 0.06 * Defense Score
+ 0.05 * Environment Difficulty Score
+ 0.04 * Stage Weighted Career Value
+ 0.02 * Longevity Score
+ Career Stage Bonus
```

## 5. 生涯总数据降权，生涯场均升权

上一版对 WS、VORP、总分钟、生涯累计价值过于敏感。本版新增：

```text
Career Per Game Score =
0.46 * PPG Score
+ 0.28 * APG Score
+ 0.18 * RPG Score
+ 0.08 * Weighted BPM Score
```

生涯总数据仍然保留，但权重降低，用来表示长期贡献，不再允许“熬工龄”压过巅峰、冠军和奖项。

## 6. 机器学习/神经网络融合比例调整

为了避免机器学习继续学习到旧版累计数据偏好，本版提高规则模型权重：

```text
Final Ability = 0.60 * Rule + 0.25 * ML + 0.15 * Deep MLP
Final Legacy  = 0.60 * Rule + 0.25 * ML + 0.15 * Deep MLP
```

这样保证模型仍然满足课程要求中的机器学习/深度神经网络要求，同时不违背篮球逻辑。

## 7. 当前样例变化

本版样例中，PG 纯实力榜已经修正为：

```text
1. Magic Johnson
2. Stephen Curry
3. Isiah Thomas
4. Walt Frazier
5. Tony Parker
6. Chris Paul
```

这符合“总冠军和冠军核心证明不能低于纯累计数据”的项目设定。
