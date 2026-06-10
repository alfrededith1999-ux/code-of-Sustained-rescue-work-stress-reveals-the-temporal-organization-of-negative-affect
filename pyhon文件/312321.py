# -*- coding: utf-8 -*-
"""
非遗润心项目实施计划（甘特图）
特点：
1. 黑体/中文字体优先
2. 任务文字放在条形框内，自动换行 + 自动缩小字号，尽量不超出
3. 分 section 排版整齐，适合申报书截图
4. 直接运行即可导出高清 PNG

如本机无“SimHei/黑体”，会自动尝试其他中文字体。
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import textwrap

# =========================
# 1. 字体与全局样式
# =========================
plt.rcParams["font.sans-serif"] = [
    "SimHei",              # 黑体
    "Microsoft YaHei",     # 微软雅黑
    "Noto Sans CJK SC",    # 思源黑体
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

TITLE = "非遗润心项目实施计划"

# =========================
# 2. 原始数据
# =========================
sections = [
    {
        "name": "启动与准备",
        "tasks": [
            ("项目启动与伦理审批",       "2026-05-01", "2026-05-31"),
            ("工具与课程共创/打样",       "2026-05-15", "2026-06-30"),
        ]
    },
    {
        "name": "实施与评估",
        "tasks": [
            ("样本招募与基线测评",         "2026-06-15", "2026-07-15"),
            ("三层级活动实施（轮1）",       "2026-07-01", "2026-08-15"),
            ("安全随访与过程监测",         "2026-07-01", "2026-09-15"),
        ]
    },
    {
        "name": "分析与产出",
        "tasks": [
            ("数据清洗与初步分析",         "2026-08-15", "2026-10-15"),
            ("机制模型与深度分析",         "2026-10-01", "2026-12-31"),
            ("论文/报告/资源包撰写",       "2026-12-01", "2027-03-15"),
            ("推广与教师培训（示范）",     "2027-02-01", "2027-04-15"),
            ("结题与成果提交",             "2027-04-01", "2027-04-30"),
        ]
    }
]

# =========================
# 3. 日期处理
# =========================
def d(s):
    return datetime.strptime(s, "%Y-%m-%d")

all_dates = []
for sec in sections:
    for _, s, e in sec["tasks"]:
        all_dates.extend([d(s), d(e)])

xmin = min(all_dates) - timedelta(days=10)
xmax = max(all_dates) + timedelta(days=10)

# =========================
# 4. 构造行（含 section 标题行）
# =========================
rows = []
for sec in sections:
    rows.append({"type": "section", "label": sec["name"]})
    for task_name, start, end in sec["tasks"]:
        rows.append({
            "type": "task",
            "label": task_name,
            "start": d(start),
            "end": d(end)
        })

n_rows = len(rows)

# =========================
# 5. 画布
# =========================
fig, ax = plt.subplots(figsize=(18, 9), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# 颜色方案（大气、正式）
section_fill = "#D9E2F3"   # section 行底色
section_edge = "#7F9DB9"

bar_fill_1 = "#8FAADC"
bar_fill_2 = "#A9C4EB"
bar_fill_3 = "#C5D9F1"
bar_edge = "#4F81BD"

grid_color = "#D9D9D9"
text_color = "black"

# =========================
# 6. 坐标轴基础设置
# =========================
ax.set_xlim(mdates.date2num(xmin), mdates.date2num(xmax))
ax.set_ylim(-0.5, n_rows - 0.5)
ax.invert_yaxis()

# 月度刻度
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.tick_top()

for label in ax.get_xticklabels():
    label.set_fontsize(11)
    label.set_fontweight("bold")
    label.set_color("black")

# 网格
ax.grid(axis="x", color=grid_color, linestyle="--", linewidth=0.8, alpha=0.8)
ax.set_axisbelow(True)

# 去掉 y 轴刻度
ax.set_yticks([])

# 边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 标题
ax.set_title(TITLE, fontsize=20, fontweight="bold", color="black", pad=28)

# =========================
# 7. 自动让文字尽量待在框内
# =========================
def fit_text_in_rect(ax, rect_patch, text_str,
                     max_fontsize=12, min_fontsize=7,
                     line_spacing=1.05):
    """
    在矩形框内放文字：
    - 自动尝试换行
    - 自动缩小字号
    - 尽量保证不超出框
    """
    fig = ax.figure

    x = rect_patch.get_x()
    y = rect_patch.get_y()
    w = rect_patch.get_width()
    h = rect_patch.get_height()

    # 先把数据坐标转为像素，估计可容纳字符数
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x + w, y + h))
    width_px = abs(p1[0] - p0[0])
    height_px = abs(p1[1] - p0[1])

    # 中文宽度粗估：每个字约等于 0.9~1.0 个字号宽
    # 根据可用像素宽度先估一个每行最大字符数
    for fs in range(max_fontsize, min_fontsize - 1, -1):
        # 经验估计：中文每字符约 fs * 1.0 像素占比系数
        # 这里保守一点，防止超框
        approx_chars_per_line = max(2, int(width_px / (fs * 1.15)))

        wrapped = textwrap.fill(
            text_str,
            width=approx_chars_per_line,
            break_long_words=False,
            break_on_hyphens=False
        )

        txt = ax.text(
            x + w / 2,
            y + h / 2,
            wrapped,
            ha="center",
            va="center",
            fontsize=fs,
            color=text_color,
            fontweight="bold",
            linespacing=line_spacing,
            clip_on=True,
            zorder=5
        )

        # 需要 draw 后才能拿到 bbox
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tb = txt.get_window_extent(renderer=renderer)

        # 给一点安全边距
        if tb.width <= width_px * 0.92 and tb.height <= height_px * 0.84:
            return txt
        else:
            txt.remove()

    # 如果最小字号仍塞不下，就强制放进去（最小字号）
    approx_chars_per_line = max(2, int(width_px / (min_fontsize * 1.2)))
    wrapped = textwrap.fill(
        text_str,
        width=approx_chars_per_line,
        break_long_words=False,
        break_on_hyphens=False
    )
    txt = ax.text(
        x + w / 2,
        y + h / 2,
        wrapped,
        ha="center",
        va="center",
        fontsize=min_fontsize,
        color=text_color,
        fontweight="bold",
        linespacing=line_spacing,
        clip_on=True,
        zorder=5
    )
    return txt

# =========================
# 8. 绘制
# =========================
bar_height = 0.72
section_height = 0.76

task_color_cycle = [bar_fill_1, bar_fill_2, bar_fill_3]
task_index = 0

for y, row in enumerate(rows):
    if row["type"] == "section":
        # section 行：横跨全图宽度
        x0 = mdates.date2num(xmin)
        w = mdates.date2num(xmax) - mdates.date2num(xmin)
        rect = Rectangle(
            (x0, y - section_height / 2),
            w,
            section_height,
            facecolor=section_fill,
            edgecolor=section_edge,
            linewidth=1.4,
            zorder=1
        )
        ax.add_patch(rect)
        fit_text_in_rect(ax, rect, row["label"], max_fontsize=13, min_fontsize=9)
    else:
        start = mdates.date2num(row["start"])
        end = mdates.date2num(row["end"])
        width = end - start + 1  # 包含截止日
        color = task_color_cycle[task_index % len(task_color_cycle)]
        task_index += 1

        rect = Rectangle(
            (start, y - bar_height / 2),
            width,
            bar_height,
            facecolor=color,
            edgecolor=bar_edge,
            linewidth=1.4,
            zorder=3
        )
        ax.add_patch(rect)

        fit_text_in_rect(ax, rect, row["label"], max_fontsize=12, min_fontsize=7)

# =========================
# 9. 左侧补充一个“阶段/任务”标签区（更整齐）
# =========================
# 这里用注释方式说明，不额外占图内空间，保持大气简洁
ax.text(
    mdates.date2num(xmin),
    -0.95,
    "阶段 / 任务",
    fontsize=11,
    fontweight="bold",
    color="black",
    ha="left",
    va="bottom"
)

# =========================
# 10. 导出
# =========================
plt.tight_layout()
plt.savefig("非遗润心项目实施计划_甘特图.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
