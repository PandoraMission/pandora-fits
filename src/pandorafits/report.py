"""Mixins for report generation"""

# Future
from __future__ import annotations

# Third-party
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _plot_table(ax, df):
    """Render a compact DataFrame table sized exactly to content."""
    ax.axis("off")
    if df.empty:
        return
    df_str = df.astype(str)
    table = ax.table(
        cellText=df_str.values,
        rowLabels=list(df_str.index),
        colLabels=list(df_str.columns),
        loc="upper left",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Size every column to fit its widest cell (axes-fraction coords)
    table.auto_set_column_width([-1, 0, 1])
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.015
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")
    table.scale(1.0, 1.1)


class ReportMixins:
    def plot_description(self, ax=None):
        """Standalone single-axes description table."""
        if ax is None:
            _, ax = plt.subplots()
        _plot_table(ax, self.describe())
        return ax

    def _build_report_figure(self, *, dpi: int = 150, wspace: float = 1.28, hspace: float = 0.2):
        """
        5-row x 6-col GridSpec report. figsize (12, 10) gives equal 2-inch squares.

        Layout
        ------
        rows 0-1, cols 0-2  description table (first half of rows)
        rows 0-1, cols 3-5  description table (overflow)
        row  2,   cols 0-1  star field image
        row  2,   cols 2-3  position - mean-position time series
        remainder           blank
        """
        fig = plt.figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(5, 6, wspace=wspace, hspace=hspace)

        ax_t1    = fig.add_subplot(gs[0:1, 0:3])
        ax_t2    = fig.add_subplot(gs[0:1, 3:6])
        ax_star  = fig.add_subplot(gs[1:3,   0:2])
        ax_astro = fig.add_subplot(gs[1:3,   2:4])
        ax_bad   = fig.add_subplot(gs[1:3,   4:6])

        materials = self.get_report_materials()

        title = materials.get("title", "")
        subtitle = materials.get("subtitle", "")
        if title:
            # Shrink the gridspec rect to leave a strip at the top for the title
            fig.get_layout_engine().set(rect=[0, 0, 1, 0.93])
            x0, y = 0.04, 0.975
            title_fs = 16
            title_use = ""
            if title != "":
                title_use += title
            if subtitle != "":
                # Just put the subtitle on the same line.
                title_use += f" :: {subtitle}"
            fig.text(x0, y, title_use, ha="left", va="top",
                     fontsize=title_fs, fontweight="bold")

        df = materials.get("tables")
        if df is not None and not df.empty:
            mid = (len(df) + 1) // 2
            _plot_table(ax_t1, df.iloc[:mid])
            _plot_table(ax_t2, df.iloc[mid:])
        else:
            ax_t1.axis("off")
            ax_t2.axis("off")

        for key, ax in [("star_field", ax_star), ("astrometry", ax_astro), ("bad_pixels", ax_bad)]:
            fn = materials.get(key)
            if callable(fn):
                fn(ax=ax)
            else:
                ax.axis("off")

        return fig

    def make_report(self, *, force: bool = False):
        """Create and cache the report figure."""
        if not force and getattr(self, "_report_fig", None) is not None:
            return self._report_fig
        self._report_fig = self._build_report_figure()
        return self._report_fig

    def get_report(self, *, force: bool = False):
        """Show the report in an interactive session (e.g. Jupyter)."""
        return self.make_report(force=force)

    def save_report(self, path: str, *, force: bool = False):
        """Save the report to a PDF."""
        fig = self.make_report(force=force)
        with PdfPages(path) as pdf:
            pdf.savefig(fig)
