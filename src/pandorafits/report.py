"""Mixins for report generation"""

# Future
from __future__ import annotations

# Third-party
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_table(ax, df):
    """Render a compact DataFrame table sized to the content."""
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

    def get_report_materials(self, *args, **kwargs):
        """Details that make up a report. This is overridden by sub classes."""
        return dict()

    def build_report_figure(
        self, *, dpi: int = 150, wspace: float = 1.3, hspace: float = 0.2
    ):
        """
        6-row x 6-col GridSpec report. figsize (12, 10) gives equal 2-inch squares.
        """
        fig = plt.figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(6, 6, wspace=wspace, hspace=hspace)

        # Create figure axes. Could do this programmatically but want finer control because they
        # are not all the same size/shape.
        report_plot_axes = list()
        report_plot_axes.append(
            fig.add_subplot(gs[0:2, 0:3])
        )  # Index 0 is reserved for table-based metrics.
        report_plot_axes.append(
            fig.add_subplot(gs[0:2, 3:6])
        )  # Index 1 is reserved for table-based metrics.
        report_plot_axes.append(fig.add_subplot(gs[2:4, 0:2]))
        report_plot_axes.append(fig.add_subplot(gs[2:4, 2:4]))
        report_plot_axes.append(fig.add_subplot(gs[2:4, 4:6]))
        report_plot_axes.append(fig.add_subplot(gs[4:6, 0:2]))
        report_plot_axes.append(fig.add_subplot(gs[4:6, 2:4]))
        report_plot_axes.append(fig.add_subplot(gs[4:6, 4:6]))

        # Overridden by sub classes
        materials = self.get_report_materials()
        if materials is None:
            # Probably being called by the wrong class. Break early.
            return fig
        if len(materials) == 0:
            return fig

        # Basic information about the target and observation time
        title = materials.get("title", "")
        subtitle = materials.get("subtitle", "")
        if title:
            # Shrink the gridspec rect to leave a strip at the top for the title
            fig_layout_engine = fig.get_layout_engine()
            if fig_layout_engine is not None:
                fig_layout_engine.set(rect=[0, 0, 1, 0.93])
            x0, y = 0.04, 0.975
            title_fs = 16
            title_use = ""
            if title != "":
                title_use += title
            if subtitle != "":
                # Just put the subtitle on the same line.
                title_use += f" :: {subtitle}"
            fig.text(
                x0,
                y,
                title_use,
                ha="left",
                va="top",
                fontsize=title_fs,
                fontweight="bold",
            )

        # Metrics about the observation as a whole.
        report_metrics = materials.get("report_metrics", None)
        if report_metrics is not None and not report_metrics.empty:
            mid = (len(report_metrics) + 1) // 2
            plot_table(report_plot_axes[0], report_metrics.iloc[:mid])
            plot_table(report_plot_axes[1], report_metrics.iloc[mid:])
        else:
            report_plot_axes[0].axis("off")
            report_plot_axes[1].axis("off")

        # Plots that are generally timeseries of one or metrics
        report_plot_funcs = materials.get("report_plots", [])
        for fig_index, ax in enumerate(report_plot_axes[2:]):

            # Plot to this axis if there is a function to plot at this location.
            if fig_index < len(report_plot_funcs):
                plot_func = report_plot_funcs[fig_index]
                if plot_func is not None:
                    plot_func(ax=ax)
                    continue
            # Otherwise turn the axis off.
            ax.axis("off")

        return fig

    def make_report(self, *, force: bool = False):
        """Create and cache the report figure."""
        if not force and getattr(self, "_report_fig", None) is not None:
            return self._report_fig
        self._report_fig = self.build_report_figure()
        return self._report_fig

    def get_report(self, *, force: bool = False):
        """Show the report in an interactive session (e.g. Jupyter)."""
        return self.make_report(force=force)

    def save_report(self, path: str, *, force: bool = False):
        """Save the report to a PDF."""
        fig = self.make_report(force=force)
        with PdfPages(path) as pdf:
            pdf.savefig(fig)
