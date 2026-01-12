"""Mixins for report generation"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class ReportMixins:
    def plot_description(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        ax.axis("off")
        df = self.describe().astype(str)

        # Draw table constrained to axes bbox
        table = pd.plotting.table(
            ax,
            df,
            loc="upper left",
            cellLoc="left",
            colLoc="left",
            bbox=[0.1, 0.1, 0.95, 0.95],  # FORCE table into axes
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Column width control (relative to axes)
        widths = {
            -1: 0.18,  # index
            0: 0.22,  # key
            1: 0.60,  # value / comment
        }

        for (row, col), cell in table.get_celld().items():
            if col in widths:
                cell.set_width(widths[col])
            cell.PAD = 0.02

            # Bold header + index
            if row == 0 or col == -1:
                cell.set_text_props(weight="bold")

        # Final vertical compaction
        table.scale(1.0, 0.9)

        return ax

    def _build_report_figure_letter_3x2(self, *, dpi: int = 150):
        """
        Build a SINGLE letter-size figure laid out as a 3x2 grid.
        Materials are drawn into provided axes (preferred). If a material
        returns an Axes from its own figure, we embed that figure as an image.
        """
        materials = list(self.get_report_materials())

        # Letter size: 8.5 x 11 inches
        fig, axes = plt.subplots(
            nrows=2,
            ncols=3,
            figsize=(11, 8.5),  # LETTER LANDSCAPE
            constrained_layout=True,
            dpi=dpi,
        )
        axes = axes.ravel()

        def _embed_axes_or_figure(target_ax, obj):
            """Render an existing Axes/Figure onto target_ax as an image."""
            target_ax.axis("off")
            if isinstance(obj, plt.Axes):
                src_fig = obj.figure
            elif isinstance(obj, plt.Figure):
                src_fig = obj
            else:
                raise TypeError(f"Expected Axes/Figure, got {type(obj)!r}")

            # draw + pull RGBA buffer (no disk)
            src_fig.canvas.draw()
            w, h = src_fig.canvas.get_width_height()
            import numpy as np

            rgba = np.frombuffer(src_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
                h, w, 4
            )
            target_ax.imshow(rgba)

        # Fill grid left-to-right, top-to-bottom
        for i, ax in enumerate(axes):
            if i >= len(materials):
                ax.axis("off")
                continue
            if materials[i] is None:
                ax.axis("off")
                continue

            item = materials[i]

            # Preferred path: if item is callable that can draw into ax
            if callable(item):
                try:
                    out = item(ax=ax)
                except TypeError:
                    out = item()
                if out is None:
                    continue
                item = out

            # If item is Axes/Figure made elsewhere, embed it
            if isinstance(item, (plt.Axes, plt.Figure)):
                # If it's an Axes on THIS fig already, nothing to do
                if isinstance(item, plt.Axes) and item.figure is fig:
                    continue
                _embed_axes_or_figure(ax, item)
                continue

            # If it's something else, user should convert it to a plot method
            raise TypeError(f"Unsupported report material type: {type(item)!r}")

        return fig

    def make_report(self, *, force: bool = False):
        """
        Create and cache a single-page report figure.
        """
        if not force and getattr(self, "_report_fig", None) is not None:
            return self._report_fig
        self._report_fig = self._build_report_figure_letter_3x2()
        return self._report_fig

    def get_report(self, *, force: bool = False):
        """
        Show the single-page report in an interactive session (e.g., Jupyter).
        """
        return self.make_report(force=force)

    def save_report(self, path: str, *, force: bool = False):
        """
        Save the single-page report to a PDF.
        """
        fig = self.make_report(force=force)
        with PdfPages(path) as pdf:
            pdf.savefig(fig)
