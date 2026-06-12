# NBA 历史地位、纯实力、球队实力排名模块（冠军含金量重置版）

本模块用于生成：

- 球员历史地位总榜
- 球员纯实力总榜
- PG / SG / SF / PF / C 五位置纯实力榜
- 球队历史赛季实力榜，例如 2017 勇士 vs 1998 公牛

## 本版重点改动

本版在之前“队友强度、对手强度、季后赛路线、生涯阶段权重、球队实力榜”的基础上，新增：

1. 冠军分层：单核冠军、多核冠军、冠军老大、冠军核心、冠军轮换。
2. 奖项分层：MVP、FMVP、一阵、DPOY、二阵、三阵、防阵、全明星不同权重。
3. 生涯总数据降权，生涯场均和巅峰表现升权。
4. 非冠军季后赛轮次加分打折，冠军核心证明显著升权。
5. 规则模型权重提高，避免无冠累计型球员在纯实力榜中过高。

详细公式见：`ALGORITHM_CHANGELOG_CHAMPIONSHIP.md`。

## 运行方式

```bash
pip install -r requirements.txt
python scripts/run_full_pipeline.py --base_dir "D:/nba_data" --out_dir "./output"
```

如果只想用已经生成的 `player_career_features.csv` 重新训练：

```bash
python scripts/train_from_features.py --features_csv "./output/features/player_career_features.csv" --out_dir "./output"
```

## 查看榜单

```bash
python scripts/preview_rankings.py --out_dir "./output" --n 20
python scripts/preview_teams.py --out_dir "./output" --n 20
```

## 关键输出

```text
output/features/player_season_features.csv
output/features/player_career_features.csv
output/features/team_season_features.csv

output/outputs/ranking_legacy_overall.csv
output/outputs/ranking_ability_overall.csv
output/outputs/ranking_ability_PG.csv
output/outputs/ranking_ability_SG.csv
output/outputs/ranking_ability_SF.csv
output/outputs/ranking_ability_PF.csv
output/outputs/ranking_ability_C.csv
output/outputs/team_ranking_results.csv
```

## DeepSeek 解释建议输入字段

建议给 DeepSeek 传递以下字段，而不是让它自己判断排名：

```json
{
  "player": "Stephen Curry",
  "position_group": "PG",
  "ability_score_final": 86.62,
  "legacy_score_final": 78.39,
  "championship_score": 100,
  "championships": 4,
  "single_core_championships": 1,
  "championship_leader_seasons": 3,
  "award_score": 73,
  "career_per_game_score": 89.73,
  "peak_3_score": 95,
  "playoff_score": 100,
  "environment_difficulty_score": 70
}
```

提示词核心要求：

```text
你只能解释模型结果，不能重新排名。请根据冠军含金量、奖项、巅峰、场均、季后赛、队友/对手强度解释该球员为什么排在当前位置。
```
