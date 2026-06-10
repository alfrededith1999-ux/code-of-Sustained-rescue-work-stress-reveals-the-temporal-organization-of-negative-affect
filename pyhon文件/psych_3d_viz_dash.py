# -*- coding: utf-8 -*-
"""
psych_3d_viz_dash.py
=========================================================
本地交互式 3D 可视化（Dash + Plotly）
- 个体：3D 轨迹（X=时间/WAVE_IDX，Y=结局分数，Z=标签/风险列/任意数值）
- 群体：3D 结构（PCA/UMAP 3D embedding），颜色可选“结局/标签”

运行示例（Windows 路径注意用双反斜杠或原始字符串）：
python psych_3d_viz_dash.py --db "D:\\date\\psych_master_db_outputs_20260123_104725\\psych_master.sqlite" --table "assessment_wide_trackable_dedup_unitfilled" --port 8050

依赖：
pip install dash plotly pandas numpy scikit-learn
可选UMAP：pip install umap-learn
"""

import argparse
import sqlite3
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

# sklearn PCA（若缺失会自动 fallback）
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# 可选 UMAP
try:
    import umap
    UMAP_OK = True
except Exception:
    UMAP_OK = False

BAD_STR = {"", "nan", "none", "null", "NULL", "#NULL!", "NaN", "None"}


def wave_to_index(w):
    """24Q1 -> 24*4+1"""
    if w is None:
        return np.nan
    s = str(w).strip()
    m = re.match(r"^(\d{2})Q([1-4])$", s)
    if not m:
        return np.nan
    yy = int(m.group(1))
    q = int(m.group(2))
    return yy * 4 + q


def clean_unit(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD_STR:
        return ""
    s = re.sub(r"\s+", "", s)
    return s


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def load_table(con, table):
    df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_outcomes(df):
    priority = [
        "PHQ9_TOTAL", "GAD7_TOTAL",
        "DASS_EQ42_DEPR", "DASS_EQ42_ANXIETY", "DASS_EQ42_STRESS",
        "DASS_Dep_x2", "DASS_Anx_x2", "DASS_Str_x2",
        "SRQ20_TOTAL",
    ]
    cols = list(df.columns)
    out = []
    for c in priority:
        if c in cols:
            out.append(c)

    kw = ["PHQ", "GAD", "DASS", "DEP", "ANX", "STRESS", "SRQ", "TOTAL", "SCORE"]
    for c in cols:
        uc = c.upper()
        if any(k in uc for k in kw):
            if c not in out:
                out.append(c)

    keep = []
    for c in out:
        x = to_num(df[c])
        if x.notna().mean() >= 0.20:
            keep.append(c)
    return keep


def detect_labels(df):
    cands = []
    for c in df.columns:
        uc = c.upper()
        if uc.startswith("Y_") or "RISK" in uc or "STATE" in uc or "ALARM" in uc:
            cands.append(c)
    keep = []
    for c in cands:
        x = to_num(df[c])
        if x.notna().mean() >= 0.10:
            keep.append(c)
    return keep


def build_default_feature_sets(df):
    cols = set(df.columns)
    sets = {}

    res = [c for c in cols if any(k in c.upper() for k in ["MSPSS", "SCS_"])]
    sets["资源/支持(自动)"] = sorted(res)

    cop = [c for c in cols if "SCSQ" in c.upper() or "COP" in c.upper()]
    sets["应对(自动)"] = sorted(cop)

    le = [c for c in cols if any(k in c.upper() for k in ["LE_", "EVENT", "STRESSOR", "EXPOS"])]
    sets["压力/事件(自动)"] = sorted(le)

    block_kw = ["NAME", "PHONE", "IP", "TIME", "SUBMIT", "来源", "来源详情", "来自IP"]
    num_cols = []
    for c in df.columns:
        uc = str(c).upper()
        if any(k.upper() in uc for k in block_kw):
            continue
        if uc in ["PERSON_ID", "PERSON_KEY", "DEMO_NAME", "DEMO_PHONE", "WAVE", "UNIT__FILLED", "DEMO_UNITDEPT", "WAVE_IDX"]:
            continue
        x = to_num(df[c])
        if x.notna().mean() >= 0.30:
            num_cols.append(c)
    sets["全部数值(>=30%非缺失)"] = sorted(num_cols)

    if all(len(v) == 0 for v in sets.values()):
        sets["全部数值(兜底)"] = []

    return sets


def pca_3d(X):
    if SKLEARN_OK:
        Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        pca = PCA(n_components=3, random_state=42)
        Z = pca.fit_transform(Xs)
        return Z, {"method": "PCA", "explained_var": pca.explained_variance_ratio_.tolist()}
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Xc = np.nan_to_num(Xc, nan=0.0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :3] * S[:3]
    return Z, {"method": "PCA_SVD_FALLBACK", "explained_var": None}


def umap_3d(X):
    if not UMAP_OK:
        raise RuntimeError("umap-learn 未安装：pip install umap-learn")
    if SKLEARN_OK:
        Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    else:
        Xs = np.nan_to_num(X, nan=0.0)
    reducer = umap.UMAP(n_components=3, n_neighbors=25, min_dist=0.2, random_state=42)
    Z = reducer.fit_transform(Xs)
    return Z, {"method": "UMAP", "n_neighbors": 25, "min_dist": 0.2}


def safe_sample(df, max_points, seed=42):
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def make_dropdown_options(values):
    return [{"label": str(v), "value": str(v)} for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite 路径")
    ap.add_argument("--table", default="assessment_wide_trackable_dedup_unitfilled",
                    help="建议用 *_trackable_dedup 或 *_unitfilled 视图")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max_points", type=int, default=12000, help="群体图最大点数（抽样）")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    df = load_table(con, args.table)
    con.close()

    if "WAVE" not in df.columns or "PERSON_ID" not in df.columns:
        raise SystemExit("[FATAL] 表必须包含 WAVE 与 PERSON_ID。请用 trackable_dedup 视图。")

    df["WAVE"] = df["WAVE"].astype(str).str.strip()
    df["WAVE_IDX"] = df["WAVE"].map(wave_to_index)

    if "UNIT__FILLED" in df.columns:
        df["UNIT__FILLED"] = df["UNIT__FILLED"].map(clean_unit)
    elif "DEMO_UNITDEPT" in df.columns:
        df["UNIT__FILLED"] = df["DEMO_UNITDEPT"].map(clean_unit)
    else:
        df["UNIT__FILLED"] = ""

    outcome_cols = detect_outcomes(df)
    label_cols = detect_labels(df)  # 你这里显示 labels:0，说明该视图里没有 Y_ 列（正常）
    feature_sets = build_default_feature_sets(df)
    feature_set_names = list(feature_sets.keys())

    name_cols = [c for c in ["DEMO_NAME_CANON", "DEMO_NAME", "NAME"] if c in df.columns]
    phone_cols = [c for c in ["DEMO_PHONE_CANON", "DEMO_PHONE", "PHONE"] if c in df.columns]
    name_col = name_cols[0] if name_cols else None
    phone_col = phone_cols[0] if phone_cols else None

    wave_options = sorted(df["WAVE"].dropna().unique().tolist())
    unit_options = sorted([u for u in df["UNIT__FILLED"].dropna().unique().tolist() if u != ""])
    unit_options = unit_options[:500]

    app = Dash(__name__)
    app.title = "心理健康 3D 可视化（个体/群体）"

    app.layout = html.Div([
        html.H2("心理健康数据库：交互式 3D（个体轨迹 + 群体结构）", style={"marginBottom": "6px"}),

        html.Div([
            html.Div([
                html.Div("结局变量（Y）"),
                dcc.Dropdown(
                    id="outcome_col",
                    options=make_dropdown_options(outcome_cols),
                    value=outcome_cols[0] if outcome_cols else None,
                    clearable=False
                ),
            ], style={"width": "33%", "display": "inline-block", "paddingRight": "10px"}),

            html.Div([
                html.Div("标签/风险列（Z或颜色）"),
                dcc.Dropdown(
                    id="label_col",
                    options=make_dropdown_options(label_cols) + [{"label": "(不使用)", "value": "__NONE__"}],
                    value="__NONE__",
                    clearable=False
                ),
                html.Div("提示：你当前表里没检测到 Y_ 标签列（labels=0），可先用(不使用)；若要用标签，请用 labeled CSV 或把标签写回视图。", style={"fontSize": "12px"})
            ], style={"width": "33%", "display": "inline-block", "paddingRight": "10px"}),

            html.Div([
                html.Div("波次筛选"),
                dcc.Dropdown(
                    id="wave_filter",
                    options=make_dropdown_options(wave_options),
                    value=wave_options,
                    multi=True
                ),
            ], style={"width": "33%", "display": "inline-block"}),
        ], style={"marginBottom": "10px"}),

        dcc.Tabs(id="tabs", value="tab-individual", children=[
            dcc.Tab(label="个体：3D轨迹", value="tab-individual"),
            dcc.Tab(label="群体：3D结构", value="tab-group"),
        ]),

        html.Div(id="tab-content"),
        html.Hr(),
    ], style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial"})

    @app.callback(Output("tab-content", "children"),
                  Input("tabs", "value"))
    def render_tab(tab):
        if tab == "tab-individual":
            return html.Div([
                html.H4("个体 3D 轨迹：选择一个人（支持搜索）"),
                html.Div([
                    html.Div([
                        html.Div("搜索：姓名包含（可空）"),
                        dcc.Input(id="search_name", type="text", value="", style={"width": "100%"})
                    ], style={"width": "30%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("搜索：手机号后4位（可空）"),
                        dcc.Input(id="search_phone4", type="text", value="", style={"width": "100%"})
                    ], style={"width": "20%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("或直接输入 PERSON_ID"),
                        dcc.Input(id="search_person_id", type="text", value="", style={"width": "100%"})
                    ], style={"width": "20%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Button("查找匹配", id="btn_find", n_clicks=0, style={"marginTop": "18px"})
                    ], style={"width": "15%", "display": "inline-block"}),

                ], style={"marginBottom": "8px"}),

                html.Div([
                    html.Div("匹配到的人（选一个）"),
                    dcc.Dropdown(id="person_pick", options=[], value=None, placeholder="先点“查找匹配”"),
                ], style={"marginBottom": "10px"}),

                dcc.Graph(id="fig_individual", style={"height": "650px"}),
                html.Div(id="individual_info", style={"whiteSpace": "pre-wrap", "marginTop": "8px"})
            ])
        else:
            return html.Div([
                html.H4("群体 3D 结构：PCA/UMAP 降维（可筛单位/抽样）"),
                html.Div([
                    html.Div([
                        html.Div("单位筛选（可空=全部）"),
                        dcc.Dropdown(
                            id="unit_filter",
                            options=make_dropdown_options(unit_options),
                            value=[],
                            multi=True
                        )
                    ], style={"width": "40%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("降维方法"),
                        dcc.Dropdown(
                            id="embed_method",
                            options=[
                                {"label": "PCA（推荐，快）", "value": "PCA"},
                                {"label": "UMAP（更像簇，需安装umap-learn）", "value": "UMAP"},
                            ],
                            value="PCA",
                            clearable=False
                        )
                    ], style={"width": "20%", "display": "inline-block", "paddingRight": "10px"}),

                    html.Div([
                        html.Div("特征集合"),
                        dcc.Dropdown(
                            id="feature_set",
                            options=make_dropdown_options(feature_set_names),
                            value=feature_set_names[0],
                            clearable=False
                        )
                    ], style={"width": "40%", "display": "inline-block"}),
                ], style={"marginBottom": "8px"}),

                html.Div([
                    html.Div([
                        html.Div("最大点数（抽样）"),
                        dcc.Slider(id="max_points_slider",
                                   min=2000, max=max(2000, min(args.max_points, 30000)), step=1000,
                                   value=min(args.max_points, 12000),
                                   marks=None,
                                   tooltip={"placement": "bottom", "always_visible": True})
                    ], style={"width": "60%", "display": "inline-block", "paddingRight": "20px"}),

                    html.Div([
                        html.Div("颜色映射"),
                        dcc.Dropdown(
                            id="color_mode",
                            options=[
                                {"label": "按结局分数（outcome）", "value": "OUTCOME"},
                                {"label": "按标签/风险列（label）", "value": "LABEL"},
                                {"label": "不着色", "value": "NONE"},
                            ],
                            value="OUTCOME",
                            clearable=False
                        )
                    ], style={"width": "38%", "display": "inline-block"}),
                ], style={"marginBottom": "8px"}),

                html.Button("生成/刷新群体3D", id="btn_group", n_clicks=0),
                dcc.Graph(id="fig_group", style={"height": "700px"}),
                html.Div(id="group_info", style={"whiteSpace": "pre-wrap", "marginTop": "8px"})
            ])

    @app.callback(
        Output("person_pick", "options"),
        Output("person_pick", "value"),
        Output("individual_info", "children"),
        Input("btn_find", "n_clicks"),
        State("search_name", "value"),
        State("search_phone4", "value"),
        State("search_person_id", "value"),
        prevent_initial_call=True
    )
    def find_person(n_clicks, q_name, q_phone4, q_pid):
        q_name = (q_name or "").strip()
        q_phone4 = (q_phone4 or "").strip()
        q_pid = (q_pid or "").strip()

        dfx = df.copy()

        if q_pid:
            cand = dfx[dfx["PERSON_ID"].astype(str) == q_pid]
        else:
            cand = dfx
            if q_name and name_col:
                cand = cand[cand[name_col].astype(str).str.contains(q_name, na=False)]
            if q_phone4 and phone_col:
                cand = cand[cand[phone_col].astype(str).str.endswith(q_phone4, na=False)]

        cand = cand.sort_values("WAVE_IDX").groupby("PERSON_ID", as_index=False).tail(1)
        cand = cand.head(50)

        opts = []
        for _, r in cand.iterrows():
            pid = str(r["PERSON_ID"])
            nm = str(r[name_col]) if name_col else ""
            ph = str(r[phone_col]) if phone_col else ""
            u = str(r.get("UNIT__FILLED", ""))
            label = f"{pid} | {nm} | {ph} | {u}"
            opts.append({"label": label, "value": pid})

        info = f"匹配到 {len(opts)} 人（最多显示50）。" + (f"\n使用字段：name={name_col}, phone={phone_col}" if (name_col or phone_col) else "\n（未检测到姓名/电话列，只能用 PERSON_ID）")
        return opts, (opts[0]["value"] if opts else None), info

    @app.callback(
        Output("fig_individual", "figure"),
        Input("person_pick", "value"),
        Input("outcome_col", "value"),
        Input("label_col", "value"),
        Input("wave_filter", "value"),
    )
    def update_individual(pid, outcome_col, label_col, wave_filter):
        fig = go.Figure()
        if pid is None or outcome_col is None:
            fig.update_layout(title="请选择一个人 + 结局变量", height=650)
            return fig

        dfi = df[df["PERSON_ID"].astype(str) == str(pid)].copy()
        if wave_filter:
            dfi = dfi[dfi["WAVE"].isin(wave_filter)]
        dfi = dfi.sort_values("WAVE_IDX")

        y = to_num(dfi[outcome_col]) if outcome_col in dfi.columns else pd.Series(np.nan, index=dfi.index)

        if label_col and label_col != "__NONE__" and label_col in dfi.columns:
            z = to_num(dfi[label_col])
            z_title = label_col
        else:
            z = pd.Series(np.zeros(len(dfi)), index=dfi.index)
            z_title = "(无Z：置0)"

        fig.add_trace(go.Scatter3d(
            x=dfi["WAVE_IDX"], y=y, z=z,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(width=4),
            text=[f"WAVE={w}<br>{outcome_col}={yy}<br>{z_title}={zz}" for w, yy, zz in zip(dfi["WAVE"], y, z)],
            hoverinfo="text",
            name="trajectory"
        ))

        fig.update_layout(
            title=f"个体轨迹 3D | PERSON_ID={pid} | Y={outcome_col} | Z={z_title}",
            scene=dict(
                xaxis_title="Time (WAVE_IDX)",
                yaxis_title=outcome_col,
                zaxis_title=z_title,
            ),
            height=650,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    @app.callback(
        Output("fig_group", "figure"),
        Output("group_info", "children"),
        Input("btn_group", "n_clicks"),
        State("outcome_col", "value"),
        State("label_col", "value"),
        State("wave_filter", "value"),
        State("unit_filter", "value"),
        State("embed_method", "value"),
        State("feature_set", "value"),
        State("max_points_slider", "value"),
        State("color_mode", "value"),
        prevent_initial_call=True
    )
    def update_group(n_clicks, outcome_col, label_col, wave_filter, unit_filter, method, feature_set, max_points, color_mode):
        fig = go.Figure()
        if outcome_col is None:
            fig.update_layout(title="请先选择结局变量", height=700)
            return fig, "缺少结局变量"

        dfg = df.copy()
        if wave_filter:
            dfg = dfg[dfg["WAVE"].isin(wave_filter)]
        if unit_filter:
            dfg = dfg[dfg["UNIT__FILLED"].isin(unit_filter)]

        feats = feature_sets.get(feature_set, [])
        if not feats:
            feats = feature_sets.get("全部数值(>=30%非缺失)", [])
        if not feats:
            fig.update_layout(title="找不到可用特征列（feature_set为空）。", height=700)
            return fig, "feature_set 为空"

        X = dfg[feats].apply(to_num)
        med = X.median(numeric_only=True)
        X = X.fillna(med)

        dfg2 = dfg.copy()
        dfg2 = safe_sample(dfg2, int(max_points), seed=42)
        X2 = X.loc[dfg2.index, feats].to_numpy(dtype=float)

        meta = {}
        try:
            if method == "UMAP":
                Z, meta = umap_3d(X2)
            else:
                Z, meta = pca_3d(X2)
        except Exception as e:
            fig.update_layout(title=f"降维失败：{e}", height=700)
            return fig, f"降维失败：{e}"

        dfg2 = dfg2.copy()
        dfg2["E1"], dfg2["E2"], dfg2["E3"] = Z[:, 0], Z[:, 1], Z[:, 2]

        color_title = "none"
        color = None
        if color_mode == "OUTCOME" and outcome_col in dfg2.columns:
            color = to_num(dfg2[outcome_col])
            color_title = outcome_col
        elif color_mode == "LABEL" and label_col and label_col != "__NONE__" and label_col in dfg2.columns:
            color = to_num(dfg2[label_col])
            color_title = label_col

        hover = []
        for _, r in dfg2.iterrows():
            hover.append(
                f"PERSON_ID={r.get('PERSON_ID','')}"
                f"<br>WAVE={r.get('WAVE','')}"
                f"<br>UNIT={r.get('UNIT__FILLED','')}"
                f"<br>{outcome_col}={r.get(outcome_col,'')}"
            )

        marker = dict(size=3)
        if color is not None:
            marker.update(color=color, colorscale="Viridis", showscale=True, colorbar=dict(title=color_title))

        fig.add_trace(go.Scatter3d(
            x=dfg2["E1"], y=dfg2["E2"], z=dfg2["E3"],
            mode="markers",
            marker=marker,
            text=hover,
            hoverinfo="text",
            name="group"
        ))

        fig.update_layout(
            title=f"群体结构 3D | method={meta.get('method')} | features={feature_set} | n={len(dfg2)}",
            scene=dict(xaxis_title="E1", yaxis_title="E2", zaxis_title="E3"),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        info = []
        info.append(f"method={meta.get('method')}, n_points={len(dfg2)}, n_features={len(feats)}")
        if meta.get("explained_var") is not None:
            info.append(f"PCA explained_var_ratio: {np.round(meta['explained_var'], 4).tolist()}")
        info.append(f"color_mode={color_mode}, color={color_title}")
        info.append(f"waves={wave_filter if wave_filter else 'ALL'}, units={'ALL' if not unit_filter else len(unit_filter)}")
        return fig, "\n".join(info)

    print("================================================================================")
    print("[OK] Loaded table:", args.table)
    print("[OK] rows:", len(df), " cols:", len(df.columns))
    print("[OK] outcomes:", len(outcome_cols), " labels:", len(label_cols))
    print("[OK] feature_sets:", {k: len(v) for k, v in feature_sets.items()})
    print(f"[RUN] http://{args.host}:{args.port}")
    print("================================================================================")

    # ✅ 新版 Dash：用 app.run；兼容旧版：若没有 run 再 fallback run_server
    if hasattr(app, "run"):
        app.run(host=args.host, port=args.port, debug=False)
    else:
        app.run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
