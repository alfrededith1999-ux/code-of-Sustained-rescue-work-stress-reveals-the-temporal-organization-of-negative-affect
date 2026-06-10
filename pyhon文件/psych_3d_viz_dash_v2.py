# -*- coding: utf-8 -*-
"""
psych_3d_viz_dash_v2.py
3D交互可视化（个体/群体），自动使用“最新大单位分类”字段（coalesce 多版本字段）。
- 修复 Dash 2.x: app.run_server -> app.run
- 修复 PCA 输入含 NaN：内置 Imputer
- 不一次性加载 869 列：按需从 SQLite 读取所选列，性能更稳

运行示例：
python psych_3d_viz_dash_v2.py --db "D:\\date\\...\\psych_master.sqlite" --table "你的view名" --port 8050
"""

import argparse
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State


# ------------------------- 工具函数 -------------------------
EMPTY_TOKENS = {"", "nan", "none", "null", "#null!"}

def _trim(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s

def get_columns(con, table_or_view: str):
    # PRAGMA 对 view/table 都可用
    rows = con.execute(f"PRAGMA table_info({table_or_view})").fetchall()
    return [r[1] for r in rows]

def detect_latest_bigunit_cols(cols):
    """
    你的库里曾出现过很多版本字段名，这里按“更新 -> 更旧”排优先级。
    只要存在就纳入 coalesce 列表。
    """
    pri = [
        # 你后续如果跑了 signature 补丁，会有这个
        "BIG_UNIT_FINAL_SIG_V2",
        # 24Q4 上下文回填版（你之前跑过）
        "BIG_UNIT_FINAL_CTX_24Q4",
        # 24Q4 fix 占位/修补版
        "BIG_UNIT_FINAL_24Q4FIX",
        # routeA 系列
        "BIG_UNIT_A2", "BIG_UNIT_A", "BIG_UNIT_FINAL_A3", "BIG_UNIT_FINAL_A4",
        # v5/v4 系列
        "BIG_UNIT_V5", "BIG_UNIT_V4", "BIG_UNIT_V3", "BIG_UNIT_V2",
        # v6 drop24 系列（你 qc_v6 脚本里用过）
        "BIG_UNIT_FINAL_V6",
        # 最原始的单位字段（很多为空）
        "UNIT__FILLED", "DEMO_UNITDEPT",
    ]
    use = [c for c in pri if c in cols]
    return use

def detect_subunit_cols(cols):
    pri = [
        "SUB_UNIT_FINAL_V6",
        "SUB_UNIT_FINAL_SIG_V2",
        "SUB_UNIT_FINAL_CTX_24Q4",
        "SUB_UNIT_FINAL_24Q4FIX",
        "SUB_UNIT_A2", "SUB_UNIT_A", "SUB_UNIT_FINAL_A3", "SUB_UNIT_FINAL_A4",
        "SUB_UNIT_V5", "SUB_UNIT_V4", "SUB_UNIT_V3", "SUB_UNIT_V2",
        "UNIT__FILLED", "DEMO_UNITDEPT",
    ]
    use = [c for c in pri if c in cols]
    return use

def build_latest_unit_series(df: pd.DataFrame, candidates: list, fallback_label="未标注大单位"):
    if not candidates:
        return pd.Series([fallback_label]*len(df), index=df.index)
    out = None
    for c in candidates:
        s = df[c].astype(str).map(_trim)
        if out is None:
            out = s
        else:
            out = np.where(out == "", s, out)
            out = pd.Series(out, index=df.index)
    out = out.map(_trim)
    out = out.replace("", fallback_label)
    return out

def is_probably_numeric(series: pd.Series) -> bool:
    # 允许少量无法转数值
    s = pd.to_numeric(series, errors="coerce")
    return s.notna().mean() >= 0.6  # 60% 可转数值

def infer_feature_sets(sample_df: pd.DataFrame, id_like_cols: set):
    """
    从 sample 推断可用数值列，并按关键字分组（更稳：不依赖全库扫描缺失率）。
    """
    cols = [c for c in sample_df.columns if c not in id_like_cols]
    numeric_cols = []
    for c in cols:
        if is_probably_numeric(sample_df[c]):
            numeric_cols.append(c)

    def pick_by_kw(kws):
        return [c for c in numeric_cols if any(kw in c.upper() for kw in kws)]

    # 你原先的分类习惯（尽量兼容）
    fs = {}
    fs["全部数值(推断)"] = numeric_cols
    fs["压力/事件(自动)"] = pick_by_kw(["LE", "EVENT", "STRESS", "Z", "TRAUMA", "PTSD", "EXPO", "PRESS"])
    fs["资源/支持(自动)"] = pick_by_kw(["RES", "SUP", "SCS", "CARE", "CAP", "PSS", "SOC"])
    fs["应对(自动)"] = pick_by_kw(["COP", "COPE", "STRATEG", "RUM", "AVOID"])
    fs["量表总分(自动)"] = [c for c in numeric_cols if any(x in c.upper() for x in ["TOTAL", "SUM", "TOT", "SCORE"])]

    # 去重且保序
    for k in list(fs.keys()):
        seen = set()
        new = []
        for c in fs[k]:
            if c not in seen:
                seen.add(c)
                new.append(c)
        fs[k] = new

    return fs

def make_sql_in_list(values):
    # 安全起见仅用于本地，仍做基本转义
    vals = []
    for v in values:
        v = str(v).replace("'", "''")
        vals.append(f"'{v}'")
    return "(" + ",".join(vals) + ")"

@lru_cache(maxsize=128)
def fetch_sqlite_cached(db_path: str, table: str, cols_key: str, where_key: str):
    """
    cache 的 key 不能是 list，所以传入 cols_key/where_key 字符串
    """
    cols = cols_key.split("|")
    where = where_key
    con = sqlite3.connect(db_path)
    sql = f"SELECT {', '.join([f'\"{c}\"' for c in cols])} FROM {table} {where}"
    df = pd.read_sql_query(sql, con)
    con.close()
    return df

def fetch_df(db_path: str, table: str, cols: list, where: str):
    cols_key = "|".join(cols)
    return fetch_sqlite_cached(db_path, table, cols_key, where).copy()


# ------------------------- Dash App -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True, help="建议传你最新的 view/table（含单位字段）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--max_points", type=int, default=8000, help="个体3D最大点数（超出会抽样）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db_path = args.db
    table = args.table

    con = sqlite3.connect(db_path)
    cols = get_columns(con, table)
    con.close()

    # 关键列
    if "PERSON_ID" not in cols or "WAVE" not in cols:
        raise SystemExit(f"[FATAL] {table} 必须包含 PERSON_ID 和 WAVE")

    big_candidates = detect_latest_bigunit_cols(cols)
    sub_candidates = detect_subunit_cols(cols)

    # 给 UI 展示一个“你当前使用的大单位字段候选列表”
    big_candidates_show = big_candidates[:]

    # 取一小段样本推断特征集
    sample_cols = ["PERSON_ID", "WAVE"] + big_candidates[:2] + sub_candidates[:2]
    sample_cols = [c for c in sample_cols if c in cols]
    # 再加一些列做数值推断（避免只推断到很少列）
    extra = [c for c in cols if c not in sample_cols][:200]  # 只取前200列做推断
    sample_cols2 = sample_cols + extra
    df_sample = fetch_df(db_path, table, sample_cols2, "LIMIT 2000")

    # id/meta 类列（不做特征）
    id_like = set([c for c in cols if c.startswith("META_") or c.startswith("DEMO_")])
    id_like |= {"PERSON_ID", "WAVE"}
    for c in big_candidates + sub_candidates:
        id_like.add(c)

    feature_sets = infer_feature_sets(df_sample, id_like_cols=id_like)
    feature_set_names = list(feature_sets.keys())

    # 波次/单位选项：用轻量 SQL 拉去重值（只取 top）
    con = sqlite3.connect(db_path)
    waves = [r[0] for r in con.execute(f"SELECT DISTINCT WAVE FROM {table} ORDER BY WAVE").fetchall()]
    con.close()

    app = Dash(__name__)
    app.title = "Psych 3D Viz (BigUnit 최신)"

    app.layout = html.Div([
        html.H3("心理测评数据库：3D 交互可视化（使用最新大单位分类）"),

        html.Div([
            html.Div([
                html.Label("视图模式"),
                dcc.Dropdown(
                    id="mode",
                    options=[
                        {"label": "个体（Individual）", "value": "ind"},
                        {"label": "群体（按大单位聚合）", "value": "grp"},
                    ],
                    value="ind",
                    clearable=False
                ),

                html.Br(),
                html.Label("波次筛选（可多选）"),
                dcc.Dropdown(
                    id="waves",
                    options=[{"label": w, "value": w} for w in waves],
                    value=waves,
                    multi=True
                ),

                html.Br(),
                html.Label("特征集合（用于降维）"),
                dcc.Dropdown(
                    id="feature_set",
                    options=[{"label": k, "value": k} for k in feature_set_names],
                    value="全部数值(推断)",
                    clearable=False
                ),

                html.Br(),
                html.Label("降维方法"),
                dcc.Dropdown(
                    id="dr_method",
                    options=[
                        {"label": "PCA(3D) + 缺失值填补", "value": "pca"},
                    ],
                    value="pca",
                    clearable=False
                ),

                html.Br(),
                html.Label("颜色映射"),
                dcc.Dropdown(
                    id="color_by",
                    options=[
                        {"label": "按大单位", "value": "big"},
                        {"label": "按波次", "value": "wave"},
                        {"label": "按“结局(阈值)”", "value": "ending"},
                    ],
                    value="big",
                    clearable=False
                ),

                html.Br(),
                html.Label("阈值（结局判定：用于ending颜色）"),
                dcc.Slider(id="ending_thr", min=0, max=1, step=0.01, value=0.7,
                           marks={0:"0",0.5:"0.5",1:"1"}),

                html.Br(),
                html.Label("结局依据列（0~1 或 概率/风险分）"),
                dcc.Input(
                    id="ending_col",
                    type="text",
                    value="",
                    placeholder="例如：RISK_PROB 或 p_hat（留空则用 PCA 第一主成分归一化当作演示）",
                    style={"width": "100%"}
                ),

                html.Hr(),
                html.Div([
                    html.Div("当前检测到的大单位字段候选（越靠前越“新”）：", style={"fontSize": 12, "color": "#555"}),
                    html.Div(", ".join(big_candidates_show) if big_candidates_show else "(未检测到)", style={"fontSize": 12}),
                ], style={"padding": "8px", "background": "#f7f7f7", "borderRadius": "8px"}),

            ], style={"width": "28%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

            html.Div([
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=[
                        dcc.Graph(id="graph3d", style={"height": "80vh"}),
                        html.Pre(id="stats", style={"whiteSpace": "pre-wrap"})
                    ]
                )
            ], style={"width": "70%", "display": "inline-block", "padding": "10px"}),
        ])
    ])

    @app.callback(
        Output("graph3d", "figure"),
        Output("stats", "children"),
        Input("mode", "value"),
        Input("waves", "value"),
        Input("feature_set", "value"),
        Input("dr_method", "value"),
        Input("color_by", "value"),
        Input("ending_thr", "value"),
        Input("ending_col", "value"),
    )
    def update(mode, waves_sel, fs_name, dr_method, color_by, thr, ending_col):
        waves_sel = waves_sel or []
        if not waves_sel:
            return go.Figure(), "请至少选择一个波次"

        feats = feature_sets.get(fs_name, [])
        if len(feats) < 3:
            return go.Figure(), f"特征集合 '{fs_name}' 可用列太少（{len(feats)}），换一个试试。"

        # 只取必要列：PERSON_ID/WAVE/单位候选/特征列/（可选 ending_col）
        base_cols = ["PERSON_ID", "WAVE"]
        unit_cols_need = list({c for c in (big_candidates + sub_candidates) if c in cols})
        use_cols = base_cols + unit_cols_need + feats[:]

        if ending_col and ending_col in cols and ending_col not in use_cols:
            use_cols.append(ending_col)

        where = f"WHERE WAVE IN {make_sql_in_list(waves_sel)}"
        df = fetch_df(db_path, table, use_cols, where)

        # 构造最新大单位（coalesce）
        df["BIG_UNIT_LATEST"] = build_latest_unit_series(df, big_candidates, fallback_label="未标注大单位")
        df["SUB_UNIT_LATEST"] = build_latest_unit_series(df, sub_candidates, fallback_label="")

        # 取特征矩阵（强制数值）
        X = df[feats].apply(pd.to_numeric, errors="coerce").values

        # 个体点数控制
        rng = np.random.default_rng(args.seed)
        if mode == "ind" and len(df) > args.max_points:
            idx = rng.choice(len(df), size=args.max_points, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
            X = X[idx, :]

        # 群体模式：按 (BIG_UNIT_LATEST, WAVE) 聚合（均值）
        if mode == "grp":
            gcols = ["BIG_UNIT_LATEST", "WAVE"]
            df_num = df.copy()
            for c in feats:
                df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
            g = df_num.groupby(gcols)[feats].mean(numeric_only=True).reset_index()
            # 对应的 ending_col 也可聚合（均值）
            if ending_col and ending_col in df_num.columns:
                g_end = df_num.groupby(gcols)[ending_col].mean(numeric_only=True).reset_index()
                g = g.merge(g_end, on=gcols, how="left")

            df = g
            X = df[feats].values

        # 降维：Imputer + Scaler + PCA3
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        X_std = StandardScaler().fit_transform(X_imp)
        emb = PCA(n_components=3, random_state=args.seed).fit_transform(X_std)

        df["X"] = emb[:, 0]
        df["Y"] = emb[:, 1]
        df["Z"] = emb[:, 2]

        # 结局：如果 ending_col 存在就用它，否则用第一主成分归一化演示
        if ending_col and ending_col in df.columns:
            yraw = pd.to_numeric(df[ending_col], errors="coerce")
            yraw = yraw.fillna(yraw.median() if yraw.notna().any() else 0.0)
            yprob = yraw
        else:
            pc1 = df["X"].astype(float)
            # 归一化到 0~1
            mn, mx = float(pc1.min()), float(pc1.max())
            yprob = (pc1 - mn) / (mx - mn + 1e-9)

        df["ENDING_PROB"] = yprob
        df["ENDING"] = np.where(df["ENDING_PROB"] >= thr, "高风险(结局)", "低风险(结局)")

        # 颜色
        if color_by == "big":
            color_series = df["BIG_UNIT_LATEST"]
            title_color = "颜色=大单位"
        elif color_by == "wave":
            color_series = df["WAVE"]
            title_color = "颜色=波次"
        else:
            color_series = df["ENDING"]
            title_color = "颜色=结局(阈值)"

        # 生成图
        # 为了性能：用 category 映射到整数颜色刻度（Plotly 3D 也能用）
        cats = pd.Series(color_series).astype(str)
        cat_list = sorted(cats.unique().tolist())
        cat_to_i = {c: i for i, c in enumerate(cat_list)}
        cval = cats.map(cat_to_i).values

        hover = []
        if mode == "ind":
            for i in range(len(df)):
                hover.append(
                    f"PERSON_ID={df.loc[i,'PERSON_ID']}<br>"
                    f"WAVE={df.loc[i,'WAVE']}<br>"
                    f"BIG_UNIT={df.loc[i,'BIG_UNIT_LATEST']}<br>"
                    f"SUB_UNIT={df.loc[i,'SUB_UNIT_LATEST']}<br>"
                    f"ENDING_PROB={df.loc[i,'ENDING_PROB']:.3f}<br>"
                    f"ENDING={df.loc[i,'ENDING']}"
                )
        else:
            for i in range(len(df)):
                hover.append(
                    f"BIG_UNIT={df.loc[i,'BIG_UNIT_LATEST']}<br>"
                    f"WAVE={df.loc[i,'WAVE']}<br>"
                    f"ENDING_PROB(mean)={df.loc[i,'ENDING_PROB']:.3f}<br>"
                    f"ENDING={df.loc[i,'ENDING']}"
                )

        fig = go.Figure(data=[
            go.Scatter3d(
                x=df["X"], y=df["Y"], z=df["Z"],
                mode="markers",
                marker=dict(
                    size=4 if mode == "ind" else 7,
                    color=cval,
                    opacity=0.75
                ),
                text=hover,
                hoverinfo="text"
            )
        ])

        fig.update_layout(
            title=f"3D降维可视化 | {title_color} | 模式={'个体' if mode=='ind' else '群体'} | 波次={','.join(waves_sel)}",
            margin=dict(l=0, r=0, t=50, b=0),
            scene=dict(
                xaxis_title="Dim1",
                yaxis_title="Dim2",
                zaxis_title="Dim3",
            ),
        )

        # 统计输出
        n = len(df)
        n_big = df["BIG_UNIT_LATEST"].nunique() if "BIG_UNIT_LATEST" in df else 0
        ending_rate = (df["ENDING"] == "高风险(结局)").mean() if "ENDING" in df else 0.0
        info = (
            f"rows={n}\n"
            f"big_units={n_big}\n"
            f"feature_set={fs_name} (n_cols={len(feats)})\n"
            f"ending_col={'(auto: PC1 normalized)' if not (ending_col and ending_col in cols) else ending_col}\n"
            f"ending_thr={thr:.2f}  high_risk_rate={ending_rate:.4f}\n"
            f"bigunit_candidates_used={big_candidates_show}\n"
        )

        return fig, info

    print("=" * 80)
    print("[OK] Loaded:", table)
    print("[OK] DB:", db_path)
    print("[OK] bigunit candidates:", big_candidates_show)
    print(f"[RUN] http://{args.host}:{args.port}")
    print("=" * 80)

    # Dash 2.x 用 app.run
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
