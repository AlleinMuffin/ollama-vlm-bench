# -*- coding: utf-8 -*-
# pip install Pillow ollama pynvml
"""
Ollama 本地视觉模型桌面性能测试工具（单文件版）
================================================

运行前准备：
    1. conda activate parcel_detective
    2. pip install Pillow ollama pynvml
    3. 确保 Ollama 服务已启动：ollama serve
    4. 确保模型已下载：ollama pull qwen3-vl:4b-instruct-q4_k_m
    5. 运行：python ollama_vision_benchmark.py

功能：
    - 启动时自动读取本机 Ollama 模型，并在下拉框中切换选择
    - 选择本地图片并显示缩略图
    - 自定义 Prompt
    - 后台线程流式请求 Ollama，界面实时显示回复
    - 测量首字延迟（TTFT）、输出 Token 数、生成速度、端到端耗时
    - 记录请求前后的 NVIDIA GPU 显存占用
    - 使用核心指标卡片和 Treeview 表格结构化展示性能报告
"""

import atexit
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import ollama
import pynvml
from PIL import Image, ImageOps, ImageTk


# =============================================================================
# 全局配置
# =============================================================================
# 优先选择的模型：如果本机未安装，会自动选择模型列表中的第一个视觉模型
MODEL_NAME = "qwen3-vl:4b-instruct-q4_k_m"

# 关闭窗口时，是否立即卸载本应用使用过的 Ollama 模型。
# Ollama 默认会让模型在显存中驻留一段时间（keep_alive），
# nvmlShutdown() 只释放监控句柄，并不会卸载模型。
UNLOAD_ON_CLOSE = True

DEFAULT_PROMPT = (
    "请仔细观察这张图片，先概述整体场景，再逐一描述其中的主要物体、人物、"
    "文字、颜色和空间关系。"
)

# 界面配色
COLOR_BG = "#f3f5f9"
COLOR_PANEL = "#ffffff"
COLOR_MUTED = "#64748b"
COLOR_BORDER = "#cbd5e1"
COLOR_BLUE = "#2563eb"
COLOR_BLUE_ACTIVE = "#1d4ed8"
COLOR_GREEN = "#15803d"
COLOR_RED = "#b91c1c"
COLOR_AMBER = "#b45309"
COLOR_CONSOLE_BG = "#0f172a"
COLOR_CONSOLE_FG = "#dbeafe"


# =============================================================================
# NVML / 显存监控
# =============================================================================
_nvml_lock = threading.Lock()
_nvml_ready = False
_nvml_handle = None


def _init_nvml_locked():
    """在已持有 _nvml_lock 的前提下初始化 NVML，并缓存 0 号 GPU 句柄。"""
    global _nvml_ready, _nvml_handle

    if _nvml_ready:
        return True

    try:
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _nvml_ready = True
        return True
    except Exception:
        # 无 NVIDIA GPU、驱动缺失或 NVML 初始化失败时，不能让 GUI 崩溃
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        _nvml_ready = False
        _nvml_handle = None
        return False


def get_vram_usage():
    """
    获取 0 号 GPU 的已用显存，单位 MB。

    返回：
        float: 已用显存（MB）
        str:   无法获取时返回 "N/A"
    """
    with _nvml_lock:
        if not _init_nvml_locked():
            return "N/A"

        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            return round(memory.used / (1024.0 * 1024.0), 1)
        except Exception:
            return "N/A"


def shutdown_nvml():
    """释放 NVML 资源；可重复调用。"""
    global _nvml_ready, _nvml_handle

    with _nvml_lock:
        if _nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        _nvml_ready = False
        _nvml_handle = None


# 进程退出时兜底释放 NVML
atexit.register(shutdown_nvml)


# =============================================================================
# 通用辅助函数
# =============================================================================
def _field(obj, name, default=None):
    """同时兼容 ollama 返回的 Pydantic 对象和普通 dict。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value):
    """把数值安全地转换为 int；失败返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unload_ollama_model(model_name):
    """
    请求 Ollama 立即卸载指定模型。

    原理：向 /api/generate 发送空请求并设置 keep_alive=0。
    返回 (是否成功, 错误信息)。
    """
    try:
        # 空 prompt 不会产生有效推理，keep_alive=0 让 runner 立即退出
        ollama.generate(
            model=model_name,
            prompt="",
            keep_alive=0,
            stream=False,
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _unload_ollama_models_blocking(model_names):
    """退出时依次卸载模型；关闭阶段不再操作任何 Tk 控件。"""
    for model_name in model_names:
        _unload_ollama_model(model_name)


# =============================================================================
# 主窗口
# =============================================================================
class OllamaVisionBenchmarkApp:
    def __init__(self, root):
        self.root = root
        self.image_path = None
        self._thumb_ref = None          # 必须保留 PhotoImage 引用，防止被 GC
        self._running = False
        self._closing = False
        self._cancel_event = threading.Event()
        self._worker = None

        # 模型列表与选择状态
        self.model_var = tk.StringVar()
        self.model_count_var = tk.StringVar(value="读取中…")
        self.model_info_var = tk.StringVar(
            value="正在读取本机 Ollama 模型列表…"
        )
        self._models = []
        self._models_by_name = {}
        self._models_loading = False
        self._used_models = set()   # 本应用实际发起过请求的模型，退出时统一卸载

        self.root.title("Ollama 视觉模型性能测试")
        self.root.minsize(960, 600)
        self.root.configure(background=COLOR_BG)

        self._setup_style()
        self._build_ui()
        self._center_window()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 窗口显示后再读取模型列表，避免启动阶段卡住界面
        self.root.after(120, self._refresh_models)

    # -------------------------------------------------------------------------
    # 界面构建
    # -------------------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = ("Microsoft YaHei UI", 10)
        style.configure(".", font=base_font)
        style.configure("TFrame", background=COLOR_PANEL)
        style.configure("TLabelframe", background=COLOR_PANEL,
                        bordercolor=COLOR_BORDER, relief="solid")
        style.configure("TLabelframe.Label", background=COLOR_PANEL,
                        foreground="#334155",
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TLabel", background=COLOR_PANEL, foreground="#1e293b")
        style.configure("Title.TLabel", background=COLOR_PANEL,
                        foreground="#0f172a",
                        font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("Muted.TLabel", background=COLOR_PANEL,
                        foreground=COLOR_MUTED,
                        font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", background=COLOR_PANEL,
                        foreground=COLOR_MUTED,
                        font=("Microsoft YaHei UI", 9))
        style.configure(
            "Report.Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground="#0f172a",
            rowheight=27,
            font=("Microsoft YaHei UI", 9),
            borderwidth=0,
        )
        style.configure(
            "Report.Treeview.Heading",
            background="#e2e8f0",
            foreground="#334155",
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(8, 6),
            relief="flat",
        )
        style.map(
            "Report.Treeview",
            background=[("selected", "#dbeafe")],
            foreground=[("selected", "#0f172a")],
        )
        style.configure("Primary.TButton",
                        background=COLOR_BLUE, foreground="#ffffff",
                        font=("Microsoft YaHei UI", 10, "bold"),
                        padding=(12, 8), borderwidth=0)
        style.map("Primary.TButton",
                  background=[("active", COLOR_BLUE_ACTIVE),
                              ("disabled", "#93b4f5")],
                  foreground=[("disabled", "#eff6ff")])
        style.configure("TButton", padding=(10, 6))

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0, minsize=360)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_right_panel(main)

    def _build_left_panel(self, parent):
        left = ttk.LabelFrame(parent, text=" 操作区 ", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(7, weight=1)

        # 标题
        header = ttk.Frame(left)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Ollama 视觉模型测速",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="自动读取本机模型，可随时切换",
            style="Muted.TLabel",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

        # 模型选择
        model_header = ttk.Frame(left)
        model_header.grid(row=1, column=0, sticky="ew", pady=(12, 4))
        model_header.columnconfigure(0, weight=1)
        ttk.Label(
            model_header, text="Ollama 模型", style="TLabel",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.model_count_label = tk.Label(
            model_header,
            textvariable=self.model_count_var,
            background=COLOR_PANEL,
            foreground=COLOR_MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        self.model_count_label.grid(row=0, column=1, sticky="e", padx=(0, 6))
        self.refresh_models_btn = ttk.Button(
            model_header,
            text="刷新",
            width=5,
            command=self._refresh_models,
        )
        self.refresh_models_btn.grid(row=0, column=2, sticky="e")
        self.unload_model_btn = ttk.Button(
            model_header,
            text="卸载",
            width=5,
            state="disabled",
            command=self._unload_selected_model,
        )
        self.unload_model_btn.grid(
            row=0, column=3, sticky="e", padx=(6, 0)
        )

        self.model_combo = ttk.Combobox(
            left,
            textvariable=self.model_var,
            state="disabled",
            values=[],
            font=("Microsoft YaHei UI", 10),
        )
        self.model_combo.grid(row=2, column=0, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        self.model_info_label = tk.Label(
            left,
            textvariable=self.model_info_var,
            background=COLOR_PANEL,
            foreground=COLOR_MUTED,
            font=("Microsoft YaHei UI", 9),
            wraplength=330,
            justify="left",
            anchor="w",
        )
        self.model_info_label.grid(row=3, column=0, sticky="w", pady=(4, 0))

        # 图片选择 + 缩略图
        image_row = ttk.Frame(left)
        image_row.grid(row=4, column=0, sticky="ew", pady=(10, 6))
        image_row.columnconfigure(1, weight=1)

        self.select_btn = ttk.Button(
            image_row, text="选择图片", command=self._select_image
        )
        self.select_btn.grid(row=0, column=0, sticky="nw", padx=(0, 10))

        thumb_holder = tk.Frame(
            image_row, width=210, height=160,
            background="#e8edf5",
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )
        thumb_holder.grid(row=0, column=1, sticky="e")
        thumb_holder.grid_propagate(False)

        self.thumb_label = tk.Label(
            thumb_holder,
            text="尚未选择\n图片",
            background="#e8edf5",
            foreground="#94a3b8",
            font=("Microsoft YaHei UI", 10),
            justify="center",
        )
        self.thumb_label.place(relx=0.5, rely=0.5, anchor="center")

        self.image_info_var = tk.StringVar(
            value="支持 PNG / JPG / JPEG / WEBP / BMP / GIF"
        )
        self.image_info_label = tk.Label(
            left,
            textvariable=self.image_info_var,
            background=COLOR_PANEL,
            foreground=COLOR_MUTED,
            font=("Microsoft YaHei UI", 9),
            wraplength=330,
            justify="left",
            anchor="w",
        )
        self.image_info_label.grid(row=5, column=0, sticky="w")

        # Prompt
        prompt_header = ttk.Frame(left)
        prompt_header.grid(row=6, column=0, sticky="ew", pady=(10, 4))
        prompt_header.columnconfigure(0, weight=1)
        ttk.Label(
            prompt_header, text="Prompt", style="TLabel",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            prompt_header, text="Ctrl+Enter 开始", style="Muted.TLabel"
        ).grid(row=0, column=1, sticky="e")

        prompt_frame = ttk.Frame(left)
        prompt_frame.grid(row=7, column=0, sticky="nsew")
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)

        self.prompt_text = tk.Text(
            prompt_frame,
            height=6,
            width=1,
            wrap="word",
            font=("Microsoft YaHei UI", 10),
            background="#ffffff",
            foreground="#1e293b",
            insertbackground="#1e293b",
            relief="flat",
            padx=9,
            pady=9,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_BLUE,
        )
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)
        self.prompt_text.bind("<Control-Return>", self._on_ctrl_enter)

        prompt_scroll = ttk.Scrollbar(
            prompt_frame, orient="vertical", command=self.prompt_text.yview
        )
        prompt_scroll.grid(row=0, column=1, sticky="ns")
        self.prompt_text.configure(yscrollcommand=prompt_scroll.set)

        # 状态 + 开始按钮
        self.status_var = tk.StringVar(
            value="就绪：正在读取 Ollama 模型列表…"
        )
        self.status_label = tk.Label(
            left,
            textvariable=self.status_var,
            background=COLOR_PANEL,
            foreground=COLOR_MUTED,
            font=("Microsoft YaHei UI", 9),
            wraplength=330,
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=8, column=0, sticky="w", pady=(10, 6))

        self.start_btn = ttk.Button(
            left,
            text="正在读取模型…",
            style="Primary.TButton",
            state="disabled",
            command=self._start_test,
        )
        self.start_btn.grid(row=9, column=0, sticky="ew", ipady=3)

    def _build_right_panel(self, parent):
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        # 流式输出
        output_box = ttk.LabelFrame(right, text=" 模型流式输出 ", padding=8)
        output_box.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        output_box.columnconfigure(0, weight=1)
        output_box.rowconfigure(0, weight=1)

        self.output_text = ScrolledText(
            output_box,
            wrap="word",
            width=1,
            state="disabled",
            font=("Microsoft YaHei UI", 11),
            background=COLOR_CONSOLE_BG,
            foreground=COLOR_CONSOLE_FG,
            insertbackground="#ffffff",
            selectbackground="#334155",
            relief="flat",
            padx=12,
            pady=10,
            spacing1=2,
            spacing3=2,
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")

        # 性能报告：顶部核心指标卡片 + 下方结构化表格
        report_box = ttk.LabelFrame(right, text=" 性能报告 ", padding=8)
        report_box.grid(row=1, column=0, sticky="ew")
        report_box.columnconfigure(0, weight=1)

        # 核心指标变量（只保存数值文本，不拼接 Markdown）
        self.speed_value_var = tk.StringVar(value="—")
        self.ttft_value_var = tk.StringVar(value="—")
        self.vram_value_var = tk.StringVar(value="—")

        cards = ttk.Frame(report_box)
        cards.grid(row=0, column=0, sticky="ew")
        for column in range(3):
            cards.columnconfigure(column, weight=1, uniform="metric_cards")

        self._create_metric_card(
            cards, 0, "🚀 生成速度", self.speed_value_var, "tokens/s",
            background="#eff6ff", foreground="#1d4ed8", border="#bfdbfe",
        )
        self._create_metric_card(
            cards, 1, "⏱️ 首字延迟", self.ttft_value_var, "ms",
            background="#fff7ed", foreground="#c2410c", border="#fed7aa",
        )
        self._create_metric_card(
            cards, 2, "💾 显存占用", self.vram_value_var, "MB（测试后）",
            background="#f5f3ff", foreground="#6d28d9", border="#ddd6fe",
        )

        # 详细数据表格
        table_frame = ttk.Frame(report_box)
        table_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.report_tree = ttk.Treeview(
            table_frame,
            columns=("metric", "value"),
            show="headings",
            height=8,
            style="Report.Treeview",
            selectmode="browse",
        )
        self.report_tree.heading("metric", text="性能指标", anchor="w")
        self.report_tree.heading("value", text="数据", anchor="e")
        self.report_tree.column(
            "metric", width=220, minwidth=150, stretch=True, anchor="w"
        )
        self.report_tree.column(
            "value", width=180, minwidth=110, stretch=True, anchor="e"
        )
        self.report_tree.grid(row=0, column=0, sticky="nsew")

        report_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.report_tree.yview,
        )
        report_scroll.grid(row=0, column=1, sticky="ns")
        self.report_tree.configure(yscrollcommand=report_scroll.set)

        self.report_tree.tag_configure("odd", background="#f8fafc")
        self.report_tree.tag_configure("even", background="#ffffff")

        self._report_rows = [
            ("model", "模型名称"),
            ("prompt_tokens", "输入 Token 数"),
            ("output_tokens", "输出 Token 数"),
            ("eval_duration", "生成耗时"),
            ("prompt_duration", "输入处理耗时"),
            ("wall_time", "测试总耗时"),
            ("vram_before", "测试前显存"),
            ("vram_after", "测试后显存"),
        ]
        for index, (row_id, label) in enumerate(self._report_rows):
            self.report_tree.insert(
                "",
                "end",
                iid=row_id,
                values=(label, "—"),
                tags=("even" if index % 2 == 0 else "odd",),
            )

    def _create_metric_card(
        self,
        parent,
        column,
        title,
        value_var,
        unit,
        background,
        foreground,
        border,
    ):
        """创建一个核心指标卡片。"""
        card = tk.Frame(
            parent,
            background=background,
            highlightbackground=border,
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0, 8) if column < 2 else 0,
        )

        tk.Label(
            card,
            text=title,
            background=background,
            foreground="#475569",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w")

        value_row = tk.Frame(card, background=background)
        value_row.pack(anchor="w", pady=(2, 0))

        tk.Label(
            value_row,
            textvariable=value_var,
            background=background,
            foreground=foreground,
            font=("Microsoft YaHei UI", 20, "bold"),
        ).pack(side="left")

        tk.Label(
            value_row,
            text=unit,
            background=background,
            foreground="#64748b",
            font=("Microsoft YaHei UI", 8),
        ).pack(side="left", padx=(5, 0), pady=(8, 0))

    def _center_window(self):
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(1220, max(960, screen_w - 100))
        height = min(820, max(600, screen_h - 100))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    # -------------------------------------------------------------------------
    # Ollama 模型发现与选择
    # -------------------------------------------------------------------------
    def _set_model_info(self, text, color=COLOR_MUTED):
        self.model_info_var.set(text)
        self.model_info_label.configure(foreground=color)

    def _update_start_button(self):
        if self._running:
            self.start_btn.configure(state="disabled", text="测试中…")
        elif self._models_loading:
            self.start_btn.configure(state="disabled", text="正在读取模型…")
        elif self.model_var.get().strip():
            self.start_btn.configure(state="normal", text="开始测试")
        else:
            self.start_btn.configure(state="disabled", text="请先选择模型")

        if hasattr(self, "unload_model_btn"):
            can_unload = bool(
                self.model_var.get().strip()
                and not self._running
                and not self._models_loading
            )
            self.unload_model_btn.configure(
                state="normal" if can_unload else "disabled"
            )

    def _refresh_models(self):
        """在后台读取 ollama list，避免网络请求阻塞 Tkinter 主线程。"""
        if self._closing or self._models_loading or self._running:
            return

        self._models_loading = True
        self.refresh_models_btn.configure(state="disabled")
        self.model_combo.configure(state="disabled")
        self.model_count_var.set("读取中…")
        self._set_model_info("正在读取本机 Ollama 模型列表…", COLOR_MUTED)
        self._update_start_button()

        threading.Thread(target=self._fetch_models_worker, daemon=True).start()

    def _fetch_models_worker(self):
        """后台线程：读取模型名称、大小、量化信息和 capabilities。"""
        try:
            response = ollama.list()
            raw_models = _field(response, "models", []) or []
            models = []

            for item in raw_models:
                name = _field(item, "model", None) or _field(item, "name", None)
                if not name:
                    continue

                details = _field(item, "details", None)
                parameter_size = _field(details, "parameter_size", "") or ""
                quantization = _field(details, "quantization_level", "") or ""
                size_bytes = _field(item, "size", 0) or 0

                # show() 能提供 capabilities（如 completion/vision/tools）。
                # 旧版本 Ollama 可能没有该字段，此时保持“能力未知”。
                capabilities = None
                try:
                    show_response = ollama.show(name)
                    raw_caps = _field(show_response, "capabilities", None)
                    if raw_caps is not None:
                        capabilities = [str(cap) for cap in raw_caps]
                except Exception:
                    capabilities = None

                models.append(
                    {
                        "name": name,
                        "parameter_size": parameter_size,
                        "quantization": quantization,
                        "size_bytes": size_bytes,
                        "capabilities": capabilities,
                        "caps_known": capabilities is not None,
                        "vision": bool(capabilities and "vision" in capabilities),
                    }
                )

            self._post_ui(self._apply_models, models)
        except Exception as exc:
            self._post_ui(
                self._apply_models_error,
                f"{type(exc).__name__}: {exc}",
            )

    def _apply_models(self, models):
        """主线程：更新下拉框，并自动选择一个合适的默认模型。"""
        self._models_loading = False
        if self._closing:
            return

        self.refresh_models_btn.configure(state="normal")
        models.sort(
            key=lambda item: (
                0 if item["vision"] else 1,
                0 if "instruct" in item["name"].lower() else 1,
                item["name"].lower(),
            )
        )
        self._models = models
        self._models_by_name = {item["name"]: item for item in models}
        names = [item["name"] for item in models]
        self.model_combo["values"] = names

        if not names:
            self.model_var.set("")
            self.model_combo.configure(state="disabled")
            self.model_count_var.set("0 个")
            self._set_model_info(
                "未发现本地 Ollama 模型，请先执行 ollama pull。",
                COLOR_RED,
            )
            self._set_status(
                "未发现 Ollama 模型。请 pull 模型后点击“刷新”。",
                COLOR_RED,
            )
            self._update_start_button()
            return

        self.model_count_var.set(f"{len(names)} 个")
        previous = self.model_var.get().strip()

        if previous in names:
            selected = previous
        elif MODEL_NAME in names:
            selected = MODEL_NAME
        else:
            # 优先选择名称中带 instruct 的视觉模型，避免默认选中 thinking 版本
            vision_instruct = [
                item["name"]
                for item in models
                if item["vision"] and "instruct" in item["name"].lower()
            ]
            vision_models = [
                item["name"] for item in models if item["vision"]
            ]
            selected = (
                vision_instruct[0]
                if vision_instruct
                else (vision_models[0] if vision_models else names[0])
            )

        self.model_combo.configure(state="readonly")
        self.model_var.set(selected)
        self._on_model_selected()

        if not self.image_path:
            vision_count = sum(1 for item in models if item["vision"])
            extra = f"，其中 {vision_count} 个支持视觉" if vision_count else ""
            self._set_status(
                f"已发现 {len(names)} 个模型{extra}，请选择图片并输入 Prompt。",
                COLOR_GREEN,
            )

        self._update_start_button()

    def _apply_models_error(self, error_text):
        """主线程：模型列表读取失败时恢复控件，不影响已有列表。"""
        self._models_loading = False
        if self._closing:
            return

        short_error = (
            error_text if len(error_text) <= 180 else error_text[:180] + "…"
        )
        self.refresh_models_btn.configure(state="normal")

        if self._models:
            self.model_combo.configure(state="readonly")
            self._set_model_info(
                f"刷新失败，继续使用当前列表：{short_error}",
                COLOR_AMBER,
            )
        else:
            self.model_var.set("")
            self.model_combo.configure(state="disabled")
            self.model_count_var.set("连接失败")
            self._set_model_info(
                f"无法读取模型列表：{short_error}",
                COLOR_RED,
            )
            self._set_status(
                "无法连接 Ollama，请确认 ollama serve 已启动后点击“刷新”。",
                COLOR_RED,
            )

        self._update_start_button()

    def _on_model_selected(self, _event=None):
        """下拉框切换时刷新模型详情。"""
        name = self.model_var.get().strip()
        info = self._models_by_name.get(name)

        if not info:
            self._set_model_info("未选择模型。", COLOR_MUTED)
            self._update_start_button()
            return

        parts = []
        if info["parameter_size"]:
            parts.append(str(info["parameter_size"]))
        if info["quantization"]:
            parts.append(str(info["quantization"]))
        if info["size_bytes"]:
            try:
                parts.append(f"{int(info['size_bytes']) / 1e9:.2f} GB")
            except (TypeError, ValueError):
                pass

        if info["caps_known"]:
            if info["vision"]:
                parts.append("支持视觉")
                color = COLOR_MUTED
            else:
                parts.append("未标记视觉能力")
                color = COLOR_AMBER
        else:
            parts.append("视觉能力未知")
            color = COLOR_MUTED

        self._set_model_info(" · ".join(parts), color)
        self._update_start_button()

    def _unload_selected_model(self):
        """手动卸载当前模型，立即释放 Ollama runner 占用的显存。"""
        if self._running or self._models_loading:
            return

        model_name = self.model_var.get().strip()
        if not model_name:
            return

        self.unload_model_btn.configure(state="disabled")
        self._set_status(
            f"正在请求 Ollama 卸载 {model_name}…",
            COLOR_BLUE,
        )
        threading.Thread(
            target=self._unload_model_worker,
            args=(model_name,),
            daemon=True,
        ).start()

    def _unload_model_worker(self, model_name):
        success, error_text = _unload_ollama_model(model_name)
        self._post_ui(
            self._finish_unload,
            model_name,
            success,
            error_text,
        )

    def _finish_unload(self, model_name, success, error_text):
        if success:
            self._used_models.discard(model_name)
            self._set_status(
                f"已卸载 {model_name}，Ollama 占用的显存已释放。",
                COLOR_GREEN,
            )
        else:
            short_error = (
                error_text
                if len(error_text) <= 180
                else error_text[:180] + "…"
            )
            self._set_status(
                f"卸载 {model_name} 失败：{short_error}",
                COLOR_RED,
            )
        self._update_start_button()

    # -------------------------------------------------------------------------
    # 界面更新（所有调用均来自主线程或 _post_ui）
    # -------------------------------------------------------------------------
    def _post_ui(self, func, *args):
        """
        把后台线程的 UI 更新请求投递回 Tkinter 主线程。

        Tkinter 控件不是线程安全的，因此绝对不能在工作线程中直接操作控件。
        """
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._run_ui(func, args))
        except (tk.TclError, RuntimeError):
            # 窗口已经销毁，忽略迟到的 UI 更新
            pass

    def _run_ui(self, func, args):
        if self._closing:
            return
        try:
            func(*args)
        except tk.TclError:
            pass

    def _set_output(self, text):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        if text:
            self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _append_output(self, text):
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def _reset_report(self, model_name=None, running=False):
        """测试开始前清空报告，使用占位符而不是 Markdown 文本。"""
        table_placeholder = "…" if running else "—"
        card_placeholder = "…" if running else "N/A"
        self.speed_value_var.set(card_placeholder)
        self.ttft_value_var.set(card_placeholder)
        self.vram_value_var.set(card_placeholder)

        labels = dict(self._report_rows)
        for row_id, _label in self._report_rows:
            value = table_placeholder
            if row_id == "model" and model_name:
                value = model_name
            self.report_tree.item(
                row_id,
                values=(labels[row_id], value),
            )

    def _update_report(self, metrics):
        """
        主线程：用后台线程计算好的数值更新卡片和 Treeview。

        metrics 中的速度、延迟、耗时、显存等全部是数值或 None，
        这里只做单位与显示精度的格式化，不拼接 Markdown。
        """
        def format_int(value):
            return "N/A" if value is None else str(value)

        def format_seconds(value):
            return "N/A" if value is None else f"{value:.2f} s"

        def format_mb(value):
            return "N/A" if value is None else f"{value:.1f} MB"

        speed = metrics.get("tokens_per_sec")
        ttft = metrics.get("ttft_ms")
        vram_end = metrics.get("vram_end_mb")

        self.speed_value_var.set(
            "N/A" if speed is None else f"{speed:.2f}"
        )
        self.ttft_value_var.set(
            "N/A" if ttft is None else f"{ttft:.2f}"
        )
        self.vram_value_var.set(
            "N/A" if vram_end is None else f"{vram_end:.1f}"
        )

        rows = {
            "model": (
                "模型名称",
                metrics.get("model_name") or "N/A",
            ),
            "prompt_tokens": (
                "输入 Token 数",
                format_int(metrics.get("prompt_eval_count")),
            ),
            "output_tokens": (
                "输出 Token 数",
                format_int(metrics.get("eval_count")),
            ),
            "eval_duration": (
                "生成耗时",
                format_seconds(metrics.get("eval_duration_s")),
            ),
            "prompt_duration": (
                "输入处理耗时",
                format_seconds(metrics.get("prompt_eval_duration_s")),
            ),
            "wall_time": (
                "测试总耗时",
                format_seconds(metrics.get("wall_time_s")),
            ),
            "vram_before": (
                "测试前显存",
                format_mb(metrics.get("vram_start_mb")),
            ),
            "vram_after": (
                "测试后显存",
                format_mb(metrics.get("vram_end_mb")),
            ),
        }

        for row_id, values in rows.items():
            self.report_tree.item(row_id, values=values)

    def _set_status(self, text, color=COLOR_MUTED):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)
        # 长状态文本变短时立即重绘，避免旧文字残影
        self.status_label.update_idletasks()

    def _set_running(self, running):
        self._running = running
        if running:
            self.select_btn.configure(state="disabled")
            self.model_combo.configure(state="disabled")
            self.refresh_models_btn.configure(state="disabled")
            self.prompt_text.configure(state="disabled")
        else:
            self.select_btn.configure(state="normal")
            self.model_combo.configure(
                state="readonly" if self._models else "disabled"
            )
            self.refresh_models_btn.configure(
                state="disabled" if self._models_loading else "normal"
            )
            self.prompt_text.configure(state="normal")
        self._update_start_button()

    # -------------------------------------------------------------------------
    # 用户操作
    # -------------------------------------------------------------------------
    def _select_image(self):
        if self._running:
            return

        path = filedialog.askopenfilename(
            title="选择要测试的图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        try:
            with Image.open(path) as source:
                source.load()
                width, height = source.size

                # 按 EXIF 方向摆正后生成缩略图
                preview = ImageOps.exif_transpose(source)
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                preview.thumbnail((210, 160), resampling)
                if preview.mode not in ("RGB", "RGBA"):
                    preview = preview.convert("RGB")

                photo = ImageTk.PhotoImage(preview.copy())
        except Exception as exc:
            messagebox.showerror(
                "无法打开图片",
                f"所选文件不是有效的图片，或图片已损坏。\n\n{exc}",
            )
            return

        # 全部成功后再更新当前图片，避免失败时丢失旧图
        self.image_path = path
        self._thumb_ref = photo
        self.thumb_label.configure(image=photo, text="", background="#e8edf5")
        self.image_info_var.set(
            f"{Path(path).name}  ·  {width} × {height} px"
        )
        self._set_status("图片已加载，可以开始测试。", COLOR_GREEN)

    def _on_ctrl_enter(self, _event):
        if not self._running:
            self._start_test()
        return "break"

    def _start_test(self):
        if self._running:
            return

        if self._models_loading:
            messagebox.showinfo("请稍候", "正在读取 Ollama 模型列表，请稍后再试。")
            return

        model_name = self.model_var.get().strip()
        if not model_name:
            messagebox.showwarning(
                "缺少模型",
                "没有可用的 Ollama 模型。\n"
                "请先执行 ollama pull，然后点击“刷新”。",
            )
            return

        model_info = self._models_by_name.get(model_name)
        if (
            model_info
            and model_info["caps_known"]
            and not model_info["vision"]
        ):
            if not messagebox.askyesno(
                "模型可能不支持视觉",
                f"Ollama 未将 {model_name} 标记为视觉模型。\n"
                "继续测试可能失败，或图片会被忽略。\n\n仍要继续吗？",
            ):
                return

        if not self.image_path or not Path(self.image_path).is_file():
            messagebox.showwarning("缺少图片", "请先选择一张图片。")
            return

        prompt = self.prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("缺少 Prompt", "请输入测试用的 Prompt。")
            return

        self._cancel_event.clear()
        self._used_models.add(model_name)
        self._set_running(True)
        self._set_output("")
        self._reset_report(model_name, running=True)
        self._set_status(
            f"正在使用 {model_name} 请求模型，请稍候…",
            COLOR_BLUE,
        )

        # 网络请求与测速必须放在后台线程，避免 Tkinter 主线程假死
        self._worker = threading.Thread(
            target=self._run_test_worker,
            args=(prompt, self.image_path, model_name),
            daemon=True,
        )
        self._worker.start()

    # -------------------------------------------------------------------------
    # 后台测速逻辑
    # -------------------------------------------------------------------------
    def _run_test_worker(self, prompt, image_path, model_name):
        bench_start = time.perf_counter()
        vram_start = get_vram_usage()

        request_start = None
        first_text_time = None
        done_chunk = None
        error_text = None
        stream = None

        try:
            if self._cancel_event.is_set() or self._closing:
                return

            image_bytes = Path(image_path).read_bytes()
            if not image_bytes:
                raise ValueError("图片文件为空")

            # TTFT 从真正发起请求前开始计时（包含连接与排队时间）
            request_start = time.perf_counter()
            stream = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_bytes],
                    }
                ],
                stream=True,
            )

            for chunk in stream:
                if self._cancel_event.is_set() or self._closing:
                    break

                message = _field(chunk, "message", None)
                content = _field(message, "content", "") if message is not None else ""

                # 兼容极少数直接返回 response 字段的服务端版本
                if not content:
                    content = _field(chunk, "response", "") or ""

                if content:
                    # 首个“有效文本”chunk：忽略纯空白内容
                    if first_text_time is None and content.strip():
                        first_text_time = time.perf_counter()

                    # 投递到主线程，形成实时打字机效果
                    self._post_ui(self._append_output, content)

                if bool(_field(chunk, "done", False)):
                    # 生成速度所需元数据只从 done=True 的最终 chunk 中读取
                    done_chunk = chunk
                    break

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            self._post_ui(
                self._append_output,
                f"\n\n[请求失败] {error_text}\n",
            )
        finally:
            # 提前 break 时关闭底层流，及时释放连接
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        # 用户关闭窗口后不再刷新界面，也不继续访问 NVML
        if self._closing:
            return

        wall_time_s = time.perf_counter() - bench_start
        vram_end = get_vram_usage()

        # 首字延迟（毫秒）
        ttft_ms = None
        if request_start is not None and first_text_time is not None:
            ttft_ms = (first_text_time - request_start) * 1000.0

        # 最终 chunk 元数据
        eval_count = _as_int(_field(done_chunk, "eval_count", None))
        eval_duration_ns = _as_int(_field(done_chunk, "eval_duration", None))
        prompt_eval_count = _as_int(
            _field(done_chunk, "prompt_eval_count", None)
        )
        prompt_eval_duration_ns = _as_int(
            _field(done_chunk, "prompt_eval_duration", None)
        )

        # 严格按以下公式计算生成速度：
        # tokens_per_sec = eval_count / (eval_duration / 1e9)
        tokens_per_sec = None
        if (
            eval_count is not None
            and eval_duration_ns is not None
            and eval_duration_ns > 0
        ):
            tokens_per_sec = eval_count / (eval_duration_ns / 1e9)

        # 后台线程直接计算好数值，UI 只负责格式化与展示
        eval_duration_s = (
            eval_duration_ns / 1e9
            if eval_duration_ns is not None
            else None
        )
        prompt_eval_duration_s = (
            prompt_eval_duration_ns / 1e9
            if prompt_eval_duration_ns is not None
            else None
        )
        vram_start_mb = (
            vram_start if isinstance(vram_start, (int, float)) else None
        )
        vram_end_mb = (
            vram_end if isinstance(vram_end, (int, float)) else None
        )

        metrics = {
            "model_name": model_name,
            "ttft_ms": ttft_ms,
            "tokens_per_sec": tokens_per_sec,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "eval_duration_s": eval_duration_s,
            "prompt_eval_duration_s": prompt_eval_duration_s,
            "wall_time_s": wall_time_s,
            "vram_start_mb": vram_start_mb,
            "vram_end_mb": vram_end_mb,
        }

        if error_text:
            status_text = "测试失败，请查看上方错误信息和报告中的可用数据。"
            status_color = COLOR_RED
        elif done_chunk is None:
            status_text = "流式输出已结束，但未收到 done=True 的统计元数据。"
            status_color = COLOR_AMBER
        else:
            status_text = "测试完成。"
            status_color = COLOR_GREEN

        # 只传数值 metrics 和状态文本，不在工作线程中拼接报告字符串
        self._post_ui(
            self._finish_test, metrics, status_text, status_color
        )

    def _finish_test(self, metrics, status_text, status_color):
        self._set_running(False)
        self._update_report(metrics)
        self._set_status(status_text, status_color)
        self._worker = None

    # -------------------------------------------------------------------------
    # 关闭窗口
    # -------------------------------------------------------------------------
    def _on_close(self):
        if self._running:
            if not messagebox.askyesno(
                "确认退出",
                "测试仍在进行中，退出会中断本次测试。\n确定要退出吗？",
            ):
                return

        self._closing = True
        self._cancel_event.set()
        models_to_unload = list(self._used_models)

        try:
            self.root.destroy()
        except tk.TclError:
            pass

        # 关键：pynvml.nvmlShutdown() 只释放监控资源，不会卸载 Ollama 模型。
        # 这里用 keep_alive=0 真正让 llama-server 退出并释放显存。
        if UNLOAD_ON_CLOSE and models_to_unload:
            unload_thread = threading.Thread(
                target=_unload_ollama_models_blocking,
                args=(models_to_unload,),
                daemon=True,
            )
            unload_thread.start()
            unload_thread.join(timeout=2.5)

        shutdown_nvml()


# =============================================================================
# 程序入口
# =============================================================================
def main():
    root = tk.Tk()
    try:
        OllamaVisionBenchmarkApp(root)
        root.mainloop()
    finally:
        shutdown_nvml()
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
