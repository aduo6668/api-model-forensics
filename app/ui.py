from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .config import BUDGET_PROFILES, PRIVACY_README_TEXT, PRIVACY_UI_TEXT, default_runtime_settings
from .runner import run_analysis


class ForensicsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("API Model Forensics")
        self.root.geometry("920x760")
        self.root.minsize(860, 680)

        defaults = default_runtime_settings()
        self.api_url_var = tk.StringVar(value=defaults["base_url"])
        self.api_key_var = tk.StringVar(value=defaults["api_key"])
        self.model_var = tk.StringVar(value=defaults["model"])
        self.provider_var = tk.StringVar(value=defaults["provider_hint"])
        self.mode_var = tk.StringVar(value=defaults["mode"])
        self.last_report_path = ""

        self._build()

    def _build(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(container, text="API Model Forensics", font=("Segoe UI", 18, "bold"))
        header.pack(anchor="w")
        subheader = ttk.Label(
            container,
            text="本地运行的模型法证 MVP。输入 API 地址、API Key 和模型名，就可以运行低成本 probe，并自动生成本地 PDF 报告。",
            wraplength=820,
        )
        subheader.pack(anchor="w", pady=(6, 12))

        form = ttk.LabelFrame(container, text="检测输入", padding=12)
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)

        self._add_row(form, 0, "API 地址", ttk.Entry(form, textvariable=self.api_url_var, width=80))
        self._add_row(form, 1, "API Key", ttk.Entry(form, textvariable=self.api_key_var, show="*", width=80))
        self._add_row(form, 2, "模型名", ttk.Entry(form, textvariable=self.model_var, width=40))
        self._add_row(form, 3, "可选供应商提示", ttk.Entry(form, textvariable=self.provider_var, width=40))

        mode_box = ttk.Combobox(
            form,
            textvariable=self.mode_var,
            values=list(BUDGET_PROFILES.keys()),
            state="readonly",
            width=20,
        )
        self._add_row(form, 4, "测试档位", mode_box)

        budget_note = ttk.Label(
            form,
            text=self._budget_note(),
            foreground="#555555",
            wraplength=620,
        )
        budget_note.grid(row=5, column=1, sticky="w", pady=(8, 0))
        self.mode_var.trace_add("write", lambda *_: budget_note.config(text=self._budget_note()))

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(12, 8))
        self.run_button = ttk.Button(buttons, text="开始检测", command=self._start_run)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(buttons, text="打开上次 PDF", command=self._open_last_report).pack(side=tk.LEFT, padx=(8, 0))

        privacy = ttk.LabelFrame(container, text="隐私说明", padding=12)
        privacy.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(privacy, text=PRIVACY_UI_TEXT, wraplength=820).pack(anchor="w")

        result_frame = ttk.LabelFrame(container, text="结果", padding=12)
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(result_frame, wrap="word", height=24)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(
            "1.0",
            "等待运行...\n\n提示：默认值 `mock://openai` 可用于本地烟测；也可以改成 `mock://mixed` 看混模/路由漂移效果。",
        )
        self.result_text.config(state=tk.DISABLED)

    def _add_row(self, parent: ttk.Frame, row: int, label: str, widget: ttk.Widget) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
        widget.grid(row=row, column=1, sticky="ew", pady=6)

    def _budget_note(self) -> str:
        profile = BUDGET_PROFILES[self.mode_var.get()]
        return (
            f"{profile['label']}：预计总 token 约 {profile['estimated_total_tokens']:,}，"
            f"建议上限 {profile['max_total_tokens']:,}。默认会尽量把单次检测压在 100 万 token 红线之下很远的位置。"
        )

    def _start_run(self) -> None:
        api_url = self.api_url_var.get().strip()
        api_key = self.api_key_var.get().strip()
        claimed_model = self.model_var.get().strip()
        if not api_url or not api_key or not claimed_model:
            messagebox.showwarning("缺少输入", "请至少填写 API 地址、API Key 和模型名。")
            return

        self.run_button.config(state=tk.DISABLED)
        self._set_result("正在启动检测...\n")

        def worker() -> None:
            try:
                result = run_analysis(
                    base_url=api_url,
                    api_key=api_key,
                    claimed_model=claimed_model,
                    provider_hint=self.provider_var.get().strip(),
                    mode=self.mode_var.get(),
                    progress_cb=lambda msg: self.root.after(0, lambda: self._append_result(msg)),
                )
                self.root.after(0, lambda: self._on_run_success(result))
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: self._on_run_failure(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_success(self, result: dict[str, Any]) -> None:
        summary = result["summary"]
        probs = summary["candidate_probabilities"]
        lines = [
            "检测完成。",
            "",
            f"结论标签: {summary['verdict_label']}",
            f"总体置信度: {summary['confidence_level']}",
            "",
            "四类概率:",
            f"- 与申报模型一致: {probs['claimed_model_probability']:.1%}",
            f"- 同家族降级: {probs['same_family_downgrade_probability']:.1%}",
            f"- 其他家族: {probs['alternative_family_probability']:.1%}",
            f"- 包装/混模/未知: {probs['wrapped_or_unknown_probability']:.1%}",
            "",
            "候选结果:",
        ]
        for item in summary["top_candidates"]:
            lines.append(f"- {item['name']}: {item['probability']:.1%} ({item['kind']})")
        lines.extend(
            [
                "",
                "主要依据:",
                f"- {summary['primary_reason']}",
                f"- {summary['secondary_reason']}",
                "",
                "证据分层:",
            ]
        )
        for evidence in summary["evidence_breakdown"]:
            lines.append(f"- {evidence['label']}: {evidence['score']:.3f}")
        lines.extend(["", "主要 caveats:"])
        for caveat in summary["primary_caveats"]:
            lines.append(f"- {caveat}")
        lines.extend(
            [
                "",
                f"Summary JSON: {result['summary_json']}",
                f"Report JSON: {result['report_json']}",
                f"Normalized Outputs: {result['normalized_outputs_json']}",
                f"PDF 报告: {result['report_pdf']}",
                f"本地运行目录: {result['run_dir']}",
                "",
                PRIVACY_README_TEXT,
            ]
        )
        self.last_report_path = result["report_pdf"]
        self._set_result("\n".join(lines))
        self.run_button.config(state=tk.NORMAL)

    def _on_run_failure(self, exc: Exception) -> None:
        self._append_result(f"运行失败: {exc}")
        self.run_button.config(state=tk.NORMAL)
        messagebox.showerror("运行失败", str(exc))

    def _open_last_report(self) -> None:
        if not self.last_report_path or not os.path.exists(self.last_report_path):
            messagebox.showinfo("尚无报告", "还没有可打开的 PDF 报告。")
            return
        os.startfile(self.last_report_path)  # type: ignore[attr-defined]

    def _set_result(self, text: str) -> None:
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state=tk.DISABLED)

    def _append_result(self, text: str) -> None:
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)


def run_app() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ForensicsApp(root)
    root.mainloop()
