# -*- coding: utf-8 -*-

import argparse
import sqlite3
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import umap  # optional
    UMAP_OK = True
except Exception:
    UMAP_OK = False

EMPTY_TOKENS = {"", "nan", "none", "null", "#null!", "NULL", "None", "NaN"}


def _trim(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in EMPTY_TOKENS else s


def wave_to_index(w):
    """ '24Q1' -> 24*10 + 1, 用于排序；不规则返回很大值 """
    s = _trim(w)
    m = re.match(r"^(\d{2})Q([1-4])$", s)
    if not m:
        return 10**9
    yy = int(m.group(1))
    qq = int(m.group(2))
    return yy * 10 + qq


def get_columns(con, table):
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def detect_person_col(cols):
    pri = ["PERSON_ID", "PERSON_KEY_V6", "PERSON_KEY", "META_ID", "ID"]
    for c in pri:
        if c in cols:
            return c
    raise RuntimeError("表/视图中找不到 PERSON_ID / PERSON_KEY 等主键列。")


def detect_name_col(cols):
    pri = ["DEMO_NAME_CANON", "DEMO_NAME", "NAME", "姓名", "1\t您的姓名"]
    for c in pri:
        if c in cols:
            return c
    return None


def detect_phone_col(cols):
    pri = ["DEMO_PHONE_CANON", "DEMO_PHONE", "PHONE", "联系电话", "2\t您的联系电话"]
    for c in pri:
        if c in cols:
            return c
    return None


def detect_bigunit_candidates(cols):
    # “越新越靠前”：会按顺序 coalesce
    pri = [
        "BIG_UNIT_FINAL_SIG_V2",
        "BIG_UNIT_FINAL_V6",
        "BIG_UNIT_FINAL_CTX_24Q4",
        "BIG_UNIT_FINAL_24Q4FIX",
        "BIG_UNIT_FINAL_A4", "BIG_UNIT_FINAL_A3",
        "BIG_UNIT_A2", "BIG_UNIT_A",
        "BIG_UNIT_V5", "BIG_UNIT_V4", "BIG_UNIT_V3", "BIG_UNIT_V2",
        "BIG_UNIT_CANON", "BIG_UNIT",
        "UNIT__FILLED", "DEMO_UNITDEPT",
    ]
    return [c for c in pri if c in cols]


def detect_outcomes(cols):
    # 你的库很大，这里尽量稳：优先常见量表总分/维度
    pri = [
        "PHQ9_TOTAL", "GAD7_TOTAL",
        "DASS21_DEP", "DASS21_ANX", "DASS21_STR",
        "DASS_EQ42_DEPR", "DASS_EQ42_ANXIETY", "DASS_EQ42_STRESS",
        "SRQ20_TOTAL",
        "DEP_TOTAL", "ANX_TOTAL", "STR_TOTAL",
    ]
    out = [c for c in pri if c in cols]

    # 再用关键词扩展（但只在列名里）
    kws = ["PHQ", "GAD", "DASS", "DEP", "ANX", "STRESS", "SRQ", "TOTAL", "SCORE", "SUM"]
    for c in cols:
        uc = c.upper()
        if any(k in uc for k in kws) and c not in out:
            out.append(c)
    return out[:120]  # 列太多就截一下，避免下拉爆炸


def detect_labels(cols):
    out = []
    for c in cols:
        uc = c.upper()
        if uc.startswith("Y_") or "RISK" in uc or "ALARM" in uc or "STATE" in uc:
            out.append(c)
    return out[:120]


def make_coalesce_expr(cols_list):
    """
    生成 SQLite 表达式：
    COALESCE(NULLIF(TRIM(COALESCE(col,'')),''), NULLIF(...), ...)
    """
    if not cols_list:
        return "''"
    parts = []
    for c in cols_list:
        parts.append(f"NULLIF(TRIM(COALESCE(\"{c}\",'')),'')")
    return "COALESCE(" + ",".join(parts) + ")"


@lru_cache(maxsize=256)
def read_sql_cached(db, sql):
    con = sqlite3.connect(db)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df


def read_sql(db, sql):
    return read_sql_cached(db, sql).copy()


def safe_numeric(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return X


def build_feature_sets(sample_df, exclude_cols):
    # 数值列：>=30% 非缺失
    numeric_cols = []
    for c in sample_df.columns:
        if c in exclude_cols:
            continue
        s = pd.to_numeric(sample_df[c], errors="coerce")
        if s.notna().mean() >= 0.30:
            numeric_cols.append(c)

    def pick_by_kw(kws):
        out = []
        for c in numeric_cols:
            uc = c.upper()
            if any(k in uc for k in kws):
                out.append(c)
        return out

    sets = {
        "全部数值(>=30%非缺失)": numeric_cols,
        "压力/事件(自动)": pick_by_kw(["LE_", "EVENT", "STRESSOR", "EXPOS", "TRAUMA", "PRESS", "Z_"]),
        "资源/支持(自动)": pick_by_kw(["MSPSS", "SCS_", "SUP", "RES", "CAP", "SOC"]),
        "应对(自动)": pick_by_kw(["SCSQ", "COP", "COPE", "RUM", "AVOID"]),
        "量表总分(自动)": pick_by_kw(["TOTAL", "SUM", "SCORE", "TOT"]),
    }

    # 去空集合 & 去重
    for k in list(sets.keys()):
        seen = set()
        new = []
        for c in sets[k]:
            if c not in seen:
                seen.add(c)
                new.append(c)
        sets[k] = new
    return sets


def fit_embedder(X, method="PCA"):
    """
    X: ndarray (n_samples, n_features) with NaN allowed
    返回：(Z, meta, transformer)
      transformer: dict{imputer, scaler, model, method}
    """
    imp = SimpleImputer(strategy="median")
    X2 = imp.fit_transform(X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X3 = scaler.fit_transform(X2)

    if method == "UMAP":
        if not UMAP_OK:
            raise RuntimeError("你选了UMAP，但未安装 umap-learn：pip install umap-learn")
        model = umap.UMAP(n_components=3, n_neighbors=25, min_dist=0.2, random_state=42)
        Z = model.fit_transform(X3)
        meta = {"method": "UMAP", "n_neighbors": 25, "min_dist": 0.2}
    else:
        model = PCA(n_components=3, random_state=42)
        Z = model.fit_transform(X3)
        meta = {"method": "PCA", "explained_var_ratio": model.explained_variance_ratio_.tolist()}

    transformer = {"imputer": imp, "scaler": scaler, "model": model, "method": method}
    return Z, meta, transformer


def transform_embedder(X, transformer):
    imp = transformer["imputer"]
    scaler = transformer["scaler"]
    model = transformer["model"]
    X2 = imp.transform(X)
    X3 = scaler.transform(X2)
    return model.transform(X3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--max_points", type=int, default=12000, help="群体点云最大点数（抽样）")
    ap.add_argument("--schema_sample_n", type=int, default=2000, help="用于推断特征集合的抽样行数")
    ap.add_argument("--bigunit_col", default="", help="手动指定大单位列名（可空=自动coalesce）")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    cols = get_columns(con, args.table)
    con.close()

    person_col = detect_person_col(cols)
    wave_col = "WAVE" if "WAVE" in cols else None
    if not wave_col:
        raise RuntimeError("表/视图中必须有 WAVE 列（如 '24Q1' 这种）。")

    name_col = detect_name_col(cols)
    phone_col = detect_phone_col(cols)

    big_candidates = detect_bigunit_candidates(cols)
    if args.bigunit_col:
        if args.bigunit_col not in cols:
            raise RuntimeError(f"--bigunit_col 你填的列不存在：{args.bigunit_col}")
        big_candidates = [args.bigunit_col]

    big_expr = make_coalesce_expr(big_candidates)  # SQLite表达式
    outcome_cols = detect_outcomes(cols)
    label_cols = detect_labels(cols)

    # 用抽样推断 feature sets（避免全表扫）
    sql_sample = f"SELECT * FROM {args.table} LIMIT {int(args.schema_sample_n)}"
    sample_df = read_sql(args.db, sql_sample)

    exclude_cols = set([person_col, wave_col])
    if name_col:
        exclude_cols.add(name_col)
    if phone_col:
        exclude_cols.add(phone_col)
    for c in big_candidates:
        exclude_cols.add(c)

    feature_sets = build_feature_sets(sample_df, exclude_cols)
    feature_set_names = list(feature_sets.keys())

    # waves & big units list
    df_waves = read_sql(args.db, f"SELECT DISTINCT \"{wave_col}\" AS W FROM {args.table} ORDER BY W")
    waves_all = [str(x) for x in df_waves["W"].dropna().tolist()]

    sql_big = (
        f"SELECT DISTINCT {big_expr} AS BIG_UNIT "
        f"FROM {args.table} "
        f"WHERE TRIM(COALESCE({big_expr},''))<>'' "
        f"ORDER BY BIG_UNIT"
    )
    df_big = read_sql(args.db, sql_big)
    big_units_all = df_big["BIG_UNIT"].astype(str).map(_trim)
    big_units_all = [x for x in big_units_all.tolist() if x != ""]
    if not big_units_all:
        big_units_all = ["(未能识别任何大单位：检查bigunit列/视图)"]

    # 简单缓存：存 embedding transformer
    EMBED_CACHE = {}  # key -> transformer/meta/feats

    app = Dash(__name__)
    app.title = "心理健康 3D（大单位/个体轨迹/结局模拟）"

    def dropdown_opts(vals, maxn=800):
        vals = vals[:maxn]
        return [{"label": str(v), "value": str(v)} for v in vals]

    app.layout = html.Div([
        html.H2("心理健康数据库：交互式 3D（大单位群体 + 个体轨迹 + 结局模拟）"),
        html.Div([
            html.Div([
                html.Div("结局变量（用于显示/着色/模拟）"),
                dcc.Dropdown(
                    id="outcome_col",
                    options=dropdown_opts(outcome_cols),
                    value=outcome_cols[0] if outcome_cols else None,
                    clearable=False
                )
            ], style={"width": "34%", "display": "inline-block", "paddingRight": "10px"}),

            html.Div([
                html.Div("风险/标签列（可空）"),
                dcc.Dropdown(
                    id="label_col",
                    options=[{"label": "(不使用)", "value": "__NONE__"}] + dropdown_opts(label_cols),
                    value="__NONE__",
                    clearable=False
                ),
                html.Div("提示：如果 labels=0 说明你当前view里没Y_列；没关系，照样能跑。", style={"fontSize": "12px"})
            ], style={"width": "33%", "display": "inline-block", "paddingRight": "10px"}),

            html.Div([
                html.Div("波次筛选"),
                dcc.Dropdown(
                    id="wave_filter",
                    options=dropdown_opts(waves_all),
                    value=waves_all,
                    multi=True
                ),
            ], style={"width": "33%", "display": "inline-block"}),
        ], style={"marginBottom": "8px"}),

        dcc.Tabs(id="tabs", value="tab-group", children=[
            dcc.Tab(label="群体：按大单位看", value="tab-group"),
            dcc.Tab(label="个体：选单位→选人→看轨迹", value="tab-ind"),
            dcc.Tab(label="结局模拟：交互选项决定结局", value="tab-ending"),
        ]),
        html.Div(id="tab-content"),
        html.Hr(),
        html.Div(id="debug_info", style={"whiteSpace": "pre-wrap", "fontSize": "12px"})
    ], style={"maxWidth": "1300px", "margin": "0 auto", "fontFamily": "Arial"})

    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab):
        if tab == "tab-group":
            return html.Div([
                html.H4("群体 3D：只看“大单位”（可选单位质心 / 人员点云）"),
                html.Div([
                    html.Div([
                        html.Div("选择大单位（可多选；空=全部）"),
                        dcc.Dropdown(
                            id="group_bigunit_filter",
                            options=dropdown_opts(big_units_all),
                            value=[],
                            multi=True
                        )
                    ], style={"width": "44%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("显示模式"),
                        dcc.Dropdown(
                            id="group_mode",
                            options=[
                                {"label": "单位质心点（推荐）", "value": "CENTROID"},
                                {"label": "人员点云（更细）", "value": "PEOPLE"},
                            ],
                            value="CENTROID",
                            clearable=False
                        )
                    ], style={"width": "18%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("降维方法"),
                        dcc.Dropdown(
                            id="group_method",
                            options=[
                                {"label": "PCA（快）", "value": "PCA"},
                                {"label": "UMAP（更像簇，需umap-learn）", "value": "UMAP"},
                            ],
                            value="PCA",
                            clearable=False
                        )
                    ], style={"width": "18%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("特征集合"),
                        dcc.Dropdown(
                            id="group_feature_set",
                            options=dropdown_opts(feature_set_names),
                            value=feature_set_names[0] if feature_set_names else None,
                            clearable=False
                        )
                    ], style={"width": "20%", "display": "inline-block"}),
                ], style={"marginBottom": "8px"}),

                html.Div([
                    html.Div("最大点数（人员点云模式会抽样）"),
                    dcc.Slider(
                        id="group_max_points",
                        min=2000, max=max(2000, min(30000, args.max_points)), step=1000,
                        value=min(args.max_points, 12000),
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={"marginBottom": "8px"}),

                html.Button("生成/刷新 群体3D", id="btn_group", n_clicks=0),
                dcc.Graph(id="fig_group", style={"height": "720px"}),
                html.Div(id="group_info", style={"whiteSpace": "pre-wrap", "marginTop": "6px"})
            ])

        if tab == "tab-ind":
            return html.Div([
                html.H4("个体 3D：先选大单位 → 再选人 → 看跨波次轨迹"),
                html.Div([
                    html.Div([
                        html.Div("大单位"),
                        dcc.Dropdown(
                            id="ind_bigunit",
                            options=dropdown_opts(big_units_all),
                            value=big_units_all[0] if big_units_all else None,
                            clearable=False
                        )
                    ], style={"width": "35%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("轨迹类型"),
                        dcc.Dropdown(
                            id="traj_mode",
                            options=[
                                {"label": "指标轨迹（X=波次，Y=结局分，Z=风险/标签）", "value": "METRIC"},
                                {"label": "嵌入轨迹（把每波次映射到3D嵌入空间）", "value": "EMBED"},
                            ],
                            value="METRIC",
                            clearable=False
                        )
                    ], style={"width": "35%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("嵌入特征集合（仅“嵌入轨迹”用）"),
                        dcc.Dropdown(
                            id="ind_feature_set",
                            options=dropdown_opts(feature_set_names),
                            value=feature_set_names[0] if feature_set_names else None,
                            clearable=False
                        )
                    ], style={"width": "30%", "display": "inline-block"}),
                ], style={"marginBottom": "8px"}),

                html.Div([
                    html.Div([
                        html.Div("搜索：姓名包含（可空）"),
                        dcc.Input(id="q_name", type="text", value="", style={"width": "100%"})
                    ], style={"width": "30%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("搜索：手机号后4位（可空）"),
                        dcc.Input(id="q_phone4", type="text", value="", style={"width": "100%"})
                    ], style={"width": "25%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div(f"或直接输入 {person_col}"),
                        dcc.Input(id="q_pid", type="text", value="", style={"width": "100%"})
                    ], style={"width": "25%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Button("查找匹配", id="btn_find", n_clicks=0, style={"marginTop": "18px"})
                    ], style={"width": "18%", "display": "inline-block"}),
                ], style={"marginBottom": "8px"}),

                html.Div([
                    html.Div("匹配到的人（选一个）"),
                    dcc.Dropdown(id="person_pick", options=[], value=None, placeholder="先点“查找匹配”"),
                ], style={"marginBottom": "10px"}),

                dcc.Graph(id="fig_ind", style={"height": "720px"}),
                html.Div(id="ind_info", style={"whiteSpace": "pre-wrap", "marginTop": "6px"})
            ])

        # tab-ending
        return html.Div([
            html.H4("结局模拟（交互选项决定结局）——演示用途：不写回DB"),
            html.Div([
                html.Div([
                    html.Div("大单位"),
                    dcc.Dropdown(
                        id="end_bigunit",
                        options=dropdown_opts(big_units_all),
                        value=big_units_all[0] if big_units_all else None,
                        clearable=False
                    )
                ], style={"width": "35%", "display": "inline-block", "paddingRight": "10px"}),

                html.Div([
                    html.Div("人（先在“个体”页查到后可复制ID粘贴这里）"),
                    dcc.Input(id="end_pid", type="text", value="", style={"width": "100%"})
                ], style={"width": "35%", "display": "inline-block", "paddingRight": "10px"}),

                html.Div([
                    html.Div("阈值（把结局分转成风险等级）"),
                    dcc.Dropdown(
                        id="end_rule",
                        options=[
                            {"label": "单阈值：>=T 为高风险", "value": "ONE"},
                            {"label": "双阈值：<T1低，T1~T2中，>=T2高", "value": "TWO"},
                        ],
                        value="TWO",
                        clearable=False
                    )
                ], style={"width": "30%", "display": "inline-block"}),
            ], style={"marginBottom": "8px"}),

            html.Div([
                html.Div("T1（低/中分界）"),
                dcc.Slider(id="thr1", min=0, max=40, step=1, value=10,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),

            html.Div([
                html.Div("T2（中/高分界）"),
                dcc.Slider(id="thr2", min=0, max=60, step=1, value=15,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),

            html.Div([
                html.Div("干预强度（+ 越大越有利）"),
                dcc.Slider(id="interv", min=0, max=10, step=1, value=3,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),

            html.Div([
                html.Div("压力变化（+ 越大越不利）"),
                dcc.Slider(id="stress", min=-10, max=10, step=1, value=2,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),

            html.Div([
                html.Div("支持/资源变化（+ 越大越有利）"),
                dcc.Slider(id="support", min=-10, max=10, step=1, value=2,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),

            html.Button("生成结局", id="btn_end", n_clicks=0),
            html.Div(id="ending_text", style={"whiteSpace": "pre-wrap", "marginTop": "10px", "fontSize": "14px"})
        ])

    # --------- helper: fetch subset ----------
    def fetch_subset(waves, bigunits, need_cols, limit=None):
        wh = []
        if waves:
            ws = ",".join([f"'{str(w).replace(\"'\",\"''\")}'" for w in waves])
            wh.append(f"\"{wave_col}\" IN ({ws})")
        if bigunits:
            bu = ",".join([f"'{str(b).replace(\"'\",\"''\")}'" for b in bigunits])
            wh.append(f"TRIM(COALESCE({big_expr},''))<>'' AND {big_expr} IN ({bu})")
        where = ("WHERE " + " AND ".join(wh)) if wh else ""
        sel = ", ".join([f"\"{c}\"" for c in need_cols])
        sql = f"SELECT {sel}, {big_expr} AS BIG_UNIT__AUTO FROM {args.table} {where}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return read_sql(args.db, sql)

    # --------- group callback ----------
    @app.callback(
        Output("fig_group", "figure"),
        Output("group_info", "children"),
        Input("btn_group", "n_clicks"),
        State("outcome_col", "value"),
        State("label_col", "value"),
        State("wave_filter", "value"),
        State("group_bigunit_filter", "value"),
        State("group_mode", "value"),
        State("group_method", "value"),
        State("group_feature_set", "value"),
        State("group_max_points", "value"),
        prevent_initial_call=True
    )
    def update_group(n, outcome_col, label_col, waves, bigunits, mode, method, fset, max_points):
        fig = go.Figure()
        if not outcome_col:
            fig.update_layout(title="请先选择结局变量", height=720)
            return fig, "缺少 outcome_col"

        feats = feature_sets.get(fset, []) if fset else []
        if not feats:
            fig.update_layout(title="feature_set 为空（请换一个特征集合）", height=720)
            return fig, "feature_set 为空"

        need_cols = [person_col, wave_col, outcome_col] + feats
        if label_col and label_col != "__NONE__":
            need_cols.append(label_col)
        if name_col:
            need_cols.append(name_col)
        if phone_col:
            need_cols.append(phone_col)

        dfg = fetch_subset(waves, bigunits, list(dict.fromkeys(need_cols)), limit=None)

        # 人员点云模式时抽样
        if mode == "PEOPLE" and len(dfg) > int(max_points):
            dfg = dfg.sample(n=int(max_points), random_state=42)

        # 构建 X
        X = safe_numeric(dfg, feats).to_numpy(dtype=float)

        # fit embedding
        key = ("GROUP", tuple(sorted(waves)) if waves else ("ALL",), tuple(sorted(bigunits)) if bigunits else ("ALL",), method, fset)
        if key in EMBED_CACHE:
            transformer = EMBED_CACHE[key]["transformer"]
            meta = EMBED_CACHE[key]["meta"]
        else:
            Z, meta, transformer = fit_embedder(X, method=("UMAP" if method == "UMAP" else "PCA"))
            EMBED_CACHE[key] = {"transformer": transformer, "meta": meta, "feats": feats}
            dfg["E1"], dfg["E2"], dfg["E3"] = Z[:, 0], Z[:, 1], Z[:, 2]

        if "E1" not in dfg.columns:
            Z = transform_embedder(X, transformer)
            dfg["E1"], dfg["E2"], dfg["E3"] = Z[:, 0], Z[:, 1], Z[:, 2]

        # 颜色：优先 outcome；如果用户选 label 且存在则用 label
        color_title = outcome_col
        color_vals = pd.to_numeric(dfg.get(outcome_col), errors="coerce")

        if label_col and label_col != "__NONE__" and label_col in dfg.columns:
            # 如果用户确实想看 label，可把下面一行改成 color_vals = label
            pass

        # 汇总到单位质心
        if mode == "CENTROID":
            grp = dfg.groupby("BIG_UNIT__AUTO", dropna=False)
            cen = grp[["E1", "E2", "E3"]].mean().reset_index()
            cen["n_rows"] = grp.size().values

            # 单位平均结局（用于hover）
            ymean = grp[outcome_col].apply(lambda s: pd.to_numeric(s, errors="coerce").mean()).reset_index(name="Y_MEAN")
            cen = cen.merge(ymean, on="BIG_UNIT__AUTO", how="left")

            # 分trace（每个单位一个点，trace不必拆太细）
            fig.add_trace(go.Scatter3d(
                x=cen["E1"], y=cen["E2"], z=cen["E3"],
                mode="markers",
                marker=dict(size=np.clip(cen["n_rows"].to_numpy() / 80.0, 6, 22),
                            color=pd.to_numeric(cen["Y_MEAN"], errors="coerce"),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=f"{outcome_col}(unit mean)")),
                text=[f"BIG_UNIT={bu}<br>n_rows={nr}<br>{outcome_col}_mean={ym:.3f}"
                      for bu, nr, ym in zip(cen["BIG_UNIT__AUTO"], cen["n_rows"], cen["Y_MEAN"])],
                hoverinfo="text",
                name="units"
            ))
            title = f"群体3D（单位质心）| method={meta.get('method')} | units={len(cen)} | waves={len(waves) if waves else 'ALL'}"
        else:
            hover = []
            for _, r in dfg.iterrows():
                nm = _trim(r.get(name_col)) if name_col else ""
                ph = _trim(r.get(phone_col)) if phone_col else ""
                hover.append(
                    f"BIG_UNIT={_trim(r.get('BIG_UNIT__AUTO'))}"
                    f"<br>{person_col}={_trim(r.get(person_col))}"
                    f"<br>WAVE={_trim(r.get(wave_col))}"
                    f"<br>{outcome_col}={_trim(r.get(outcome_col))}"
                    + (f"<br>NAME={nm}" if nm else "")
                    + (f"<br>PHONE={ph}" if ph else "")
                )

            fig.add_trace(go.Scatter3d(
                x=dfg["E1"], y=dfg["E2"], z=dfg["E3"],
                mode="markers",
                marker=dict(size=3, color=color_vals, colorscale="Viridis", showscale=True,
                            colorbar=dict(title=outcome_col)),
                text=hover,
                hoverinfo="text",
                name="people"
            ))
            title = f"群体3D（人员点云）| method={meta.get('method')} | n_points={len(dfg)} | waves={len(waves) if waves else 'ALL'}"

        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="E1", yaxis_title="E2", zaxis_title="E3"),
            height=720,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        info = []
        info.append(f"大单位列（coalesce优先级）={big_candidates}")
        info.append(f"bigunit_expr={big_expr}")
        info.append(f"features({fset})={len(feats)} | mode={mode} | method={meta.get('method')}")
        if meta.get("explained_var_ratio") is not None:
            info.append(f"PCA explained_var_ratio={np.round(meta['explained_var_ratio'], 4).tolist()}")
        return fig, "\n".join(info)

    # --------- find person in unit ----------
    @app.callback(
        Output("person_pick", "options"),
        Output("person_pick", "value"),
        Output("ind_info", "children"),
        Input("btn_find", "n_clicks"),
        State("ind_bigunit", "value"),
        State("wave_filter", "value"),
        State("q_name", "value"),
        State("q_phone4", "value"),
        State("q_pid", "value"),
        prevent_initial_call=True
    )
    def find_person(n, bigunit, waves, qname, qphone4, qpid):
        qname = (qname or "").strip()
        qphone4 = (qphone4 or "").strip()
        qpid = (qpid or "").strip()

        need_cols = [person_col, wave_col]
        if name_col:
            need_cols.append(name_col)
        if phone_col:
            need_cols.append(phone_col)

        dfx = fetch_subset(waves, [bigunit] if bigunit else [], need_cols, limit=None)

        if qpid:
            dfx = dfx[dfx[person_col].astype(str) == qpid]
        else:
            if qname and name_col:
                dfx = dfx[dfx[name_col].astype(str).str.contains(qname, na=False)]
            if qphone4 and phone_col:
                dfx = dfx[dfx[phone_col].astype(str).str.endswith(qphone4, na=False)]

        # 每人取最后一波（只用于列表展示）
        dfx["_widx"] = dfx[wave_col].map(wave_to_index)
        dfx = dfx.sort_values("_widx").groupby(person_col, as_index=False).tail(1)
        dfx = dfx.head(60)

        opts = []
        for _, r in dfx.iterrows():
            pid = _trim(r.get(person_col))
            nm = _trim(r.get(name_col)) if name_col else ""
            ph = _trim(r.get(phone_col)) if phone_col else ""
            label = f"{pid} | {nm} | {ph}"
            opts.append({"label": label, "value": pid})

        info = f"单位={bigunit}\n匹配到 {len(opts)} 人（最多显示60）\n主键列={person_col} | name={name_col} | phone={phone_col}"
        val = opts[0]["value"] if opts else None
        return opts, val, info

    # --------- individual trajectory ----------
    @app.callback(
        Output("fig_ind", "figure"),
        Output("debug_info", "children"),
        Input("person_pick", "value"),
        Input("outcome_col", "value"),
        Input("label_col", "value"),
        Input("wave_filter", "value"),
        Input("traj_mode", "value"),
        Input("ind_bigunit", "value"),
        Input("ind_feature_set", "value"),
    )
    def update_ind(pid, outcome_col, label_col, waves, traj_mode, bigunit, fset):
        fig = go.Figure()
        if not pid or not outcome_col:
            fig.update_layout(title="先选一个人 + 结局变量", height=720)
            return fig, ""

        # 拉取该人所有波次（可按wave_filter）
        need_cols = [person_col, wave_col, outcome_col]
        if label_col and label_col != "__NONE__":
            need_cols.append(label_col)
        if name_col:
            need_cols.append(name_col)
        if phone_col:
            need_cols.append(phone_col)

        # person 过滤：直接 SQL where 更快
        pid_safe = str(pid).replace("'", "''")
        ws = ",".join([f"'{str(w).replace(\"'\",\"''\")}'" for w in (waves or [])])
        where = f"WHERE \"{person_col}\"='{pid_safe}'"
        if waves:
            where += f" AND \"{wave_col}\" IN ({ws})"
        sql = f"SELECT {', '.join([f'\"{c}\"' for c in need_cols])}, {big_expr} AS BIG_UNIT__AUTO FROM {args.table} {where}"
        dfi = read_sql(args.db, sql)

        dfi["_widx"] = dfi[wave_col].map(wave_to_index)
        dfi = dfi.sort_values("_widx")

        nm = _trim(dfi[name_col].iloc[0]) if (name_col and len(dfi) > 0) else ""
        ph = _trim(dfi[phone_col].iloc[0]) if (phone_col and len(dfi) > 0) else ""
        bu = _trim(dfi["BIG_UNIT__AUTO"].iloc[0]) if len(dfi) > 0 else ""

        if traj_mode == "EMBED":
            feats = feature_sets.get(fset, []) if fset else []
            if not feats:
                fig.update_layout(title="嵌入轨迹：feature_set 为空（换一个特征集合）", height=720)
                return fig, f"embed feature_set empty: {fset}"

            # 用“该单位+所选波次”的样本拟合 embedder（抽样）
            need_cols2 = [person_col, wave_col] + feats
            dfg = fetch_subset(waves, [bu] if bu else ([bigunit] if bigunit else []), need_cols2, limit=None)
            if len(dfg) > 6000:
                dfg = dfg.sample(n=6000, random_state=42)

            Xg = safe_numeric(dfg, feats).to_numpy(dtype=float)
            key = ("IND_EMBED", tuple(sorted(waves)) if waves else ("ALL",), bu or bigunit or "UNKNOWN", "PCA", fset)
            if key in EMBED_CACHE:
                transformer = EMBED_CACHE[key]["transformer"]
                meta = EMBED_CACHE[key]["meta"]
            else:
                Zg, meta, transformer = fit_embedder(Xg, method="PCA")
                EMBED_CACHE[key] = {"transformer": transformer, "meta": meta, "feats": feats}

            # transform 本人
            Xi = safe_numeric(dfi, feats).to_numpy(dtype=float)
            Zi = transform_embedder(Xi, transformer)
            dfi["E1"], dfi["E2"], dfi["E3"] = Zi[:, 0], Zi[:, 1], Zi[:, 2]

            hover = []
            for _, r in dfi.iterrows():
                hover.append(
                    f"WAVE={_trim(r.get(wave_col))}"
                    f"<br>{outcome_col}={_trim(r.get(outcome_col))}"
                    + (f"<br>{label_col}={_trim(r.get(label_col))}" if (label_col and label_col != "__NONE__") else "")
                )

            fig.add_trace(go.Scatter3d(
                x=dfi["E1"], y=dfi["E2"], z=dfi["E3"],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(width=4),
                text=hover, hoverinfo="text",
                name="embed_trajectory"
            ))
            fig.update_layout(
                title=f"嵌入轨迹 3D | {person_col}={pid} | {nm} | {ph} | BIG_UNIT={bu}",
                scene=dict(xaxis_title="E1", yaxis_title="E2", zaxis_title="E3"),
                height=720,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            dbg = f"[EMBED] feats={len(feats)} | PCA explained_var={np.round(meta.get('explained_var_ratio',[]),4).tolist() if meta.get('explained_var_ratio') else None}"
            return fig, dbg

        # 默认：METRIC
        y = pd.to_numeric(dfi[outcome_col], errors="coerce")
        if label_col and label_col != "__NONE__" and label_col in dfi.columns:
            z = pd.to_numeric(dfi[label_col], errors="coerce")
            ztitle = label_col
        else:
            z = pd.Series(np.zeros(len(dfi)), index=dfi.index)
            ztitle = "(无label：置0)"

        fig.add_trace(go.Scatter3d(
            x=dfi["_widx"], y=y, z=z,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(width=4),
            text=[f"WAVE={w}<br>{outcome_col}={yy}<br>{ztitle}={zz}"
                  for w, yy, zz in zip(dfi[wave_col], y, z)],
            hoverinfo="text",
            name="metric_trajectory"
        ))
        fig.update_layout(
            title=f"指标轨迹 3D | {person_col}={pid} | {nm} | {ph} | BIG_UNIT={bu}",
            scene=dict(xaxis_title="Time(WAVE_IDX)", yaxis_title=outcome_col, zaxis_title=ztitle),
            height=720,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig, ""

    # --------- ending simulation ----------
    @app.callback(
        Output("ending_text", "children"),
        Input("btn_end", "n_clicks"),
        State("end_bigunit", "value"),
        State("end_pid", "value"),
        State("outcome_col", "value"),
        State("end_rule", "value"),
        State("thr1", "value"),
        State("thr2", "value"),
        State("interv", "value"),
        State("stress", "value"),
        State("support", "value"),
        State("wave_filter", "value"),
        prevent_initial_call=True
    )
    def make_ending(n, bigunit, pid, outcome_col, rule, t1, t2, interv, stress, support, waves):
        pid = (pid or "").strip()
        if not pid:
            return "请在这里粘贴一个人的ID（比如 PERSON_ID）。\n建议：到“个体”页查到后复制过来。"
        if not outcome_col:
            return "请先在顶部选择一个结局变量 outcome_col。"

        pid_safe = pid.replace("'", "''")
        ws = ",".join([f"'{str(w).replace(\"'\",\"''\")}'" for w in (waves or [])])
        where = f"WHERE \"{person_col}\"='{pid_safe}'"
        if waves:
            where += f" AND \"{wave_col}\" IN ({ws})"
        # 取最后一波当“当前分数”
        sql = f"""
        SELECT "{wave_col}" AS W, "{outcome_col}" AS Y, {big_expr} AS BU
        FROM {args.table}
        {where}
        """
        dfi = read_sql(args.db, sql)
        if dfi.empty:
            return f"没在当前波次筛选里找到此人：{pid}\n你可以把波次筛选改成ALL，或确认ID是否正确。"

        dfi["_widx"] = dfi["W"].map(wave_to_index)
        dfi = dfi.sort_values("_widx")
        y0 = pd.to_numeric(dfi["Y"], errors="coerce").iloc[-1]
        w0 = dfi["W"].iloc[-1]
        bu = _trim(dfi["BU"].iloc[-1]) or bigunit or "(未知单位)"

        if pd.isna(y0):
            y0 = 0.0

        # —— 核心：交互选项决定结局（演示）——
        # 假设：干预/支持会降低分数，压力会提高分数
        # 你要“更硬核”，后续我可以把你真实特征列接进来做“基于模型的what-if”
        y_sim = float(y0) + float(stress) * 0.8 - float(support) * 0.6 - float(interv) * 0.9

        # 风险等级
        if rule == "ONE":
            risk = "高风险" if y_sim >= t1 else "低风险"
        else:
            if y_sim < t1:
                risk = "低风险"
            elif y_sim < t2:
                risk = "中风险"
            else:
                risk = "高风险"

        # 一个“分支结局文本”
        if risk == "高风险" and interv <= 2:
            ending = "结局：风险持续上行（未充分干预）。建议：提高干预强度 + 优先减少压力暴露。"
        elif risk == "高风险" and interv >= 6:
            ending = "结局：虽然仍高风险，但出现“止跌”迹象（强干预抵消部分压力）。建议：继续维持干预并增强支持。"
        elif risk == "中风险" and support >= 4:
            ending = "结局：进入可控区（支持提升带来缓冲）。建议：维持支持与应对投入，观察下一波次是否继续下降。"
        else:
            ending = "结局：整体趋稳（在当前设定下）。建议：保持规律，避免压力反弹。"

        txt = []
        txt.append(f"人：{person_col}={pid}")
        txt.append(f"单位：{bu}")
        txt.append(f"当前波次：{w0} | 当前分数 {outcome_col}={y0:.2f}")
        txt.append(f"交互设定：干预={interv} | 压力变化={stress} | 支持变化={support}")
        txt.append(f"模拟后分数：{outcome_col}_SIM={y_sim:.2f}")
        txt.append(f"风险判定：{risk}（规则={rule}，T1={t1}, T2={t2}）")
        txt.append(ending)
        txt.append("\n注：这是演示级“规则模拟”，不等价于因果推断；你要严谨版，我可以接入你训练好的模型做真正的what-if。")
        return "\n".join(txt)

    print("================================================================================")
    print("[OK] db:", args.db)
    print("[OK] table/view:", args.table)
    print("[OK] person_col:", person_col, " wave_col:", wave_col)
    print("[OK] bigunit candidates:", big_candidates)
    print("[OK] outcomes:", len(outcome_cols), " labels:", len(label_cols))
    print("[OK] feature_sets:", {k: len(v) for k, v in feature_sets.items()})
    print(f"[RUN] http://{args.host}:{args.port}")
    print("================================================================================")

    # ✅ 兼容新版 Dash：run_server 已弃用，用 run；老版本再 fallback
    if hasattr(app, "run"):
        app.run(host=args.host, port=args.port, debug=False)
    else:
        app.run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
