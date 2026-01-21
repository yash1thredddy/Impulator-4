"""Reusable UI components for IMPULATOR."""
from frontend.ui.components.charts import (
    create_scatter_plot,
    create_3d_scatter,
    create_histogram,
    create_box_plot,
    render_chart_with_viewer,
    render_chart_controls,
)
from frontend.ui.components.molecule_viewer import (
    embed_structure_viewer,
    render_structure_viewer_hint,
    prepare_chart_customdata,
    render_smiles_input,
    render_2d_structure,
)
from frontend.ui.components.structure_viewer import (
    get_structure_viewer_component,
    get_structure_viewer_hint,
)
from frontend.ui.components.sidebar import (
    render_sidebar,
    render_active_jobs_polling,
    start_polling,
    stop_polling,
    is_polling_active,
)
from frontend.ui.components.job_form import (
    render_job_form,
    render_csv_upload_form,
)
from frontend.ui.components.compound_card import (
    render_compound_card,
    render_compound_grid,
    render_compound_list,
)
from frontend.ui.components.duplicate_dialog import (
    render_duplicate_dialog,
    clear_duplicate_dialog_state,
)

__all__ = [
    # Charts
    "create_scatter_plot",
    "create_3d_scatter",
    "create_histogram",
    "create_box_plot",
    "render_chart_with_viewer",
    "render_chart_controls",
    # Structure viewer
    "embed_structure_viewer",
    "render_structure_viewer_hint",
    "prepare_chart_customdata",
    "render_smiles_input",
    "render_2d_structure",
    "get_structure_viewer_component",
    "get_structure_viewer_hint",
    # Sidebar
    "render_sidebar",
    "render_active_jobs_polling",
    "start_polling",
    "stop_polling",
    "is_polling_active",
    # Job form
    "render_job_form",
    "render_csv_upload_form",
    # Compound card
    "render_compound_card",
    "render_compound_grid",
    "render_compound_list",
    # Duplicate dialog
    "render_duplicate_dialog",
    "clear_duplicate_dialog_state",
]
