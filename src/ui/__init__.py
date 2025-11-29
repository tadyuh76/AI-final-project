"""
Giao dien nguoi dung cho SafeRoute HCM.
Cung cap cac widget PyQt6 cho truc quan hoa va dieu khien mo phong so tan.
"""

from .styles import (
    COLORS,
    ColorPalette,
    MAIN_STYLESHEET,
    Sizes,
    Animation,
    MapStyle,
    hex_to_rgb,
    rgba_from_hex,
)

from .map_widget import (
    MapWidget,
    MapCanvas,
    PopulationZoneItem,
    ShelterItem,
    HazardZoneItem,
    RouteItem,
    EvacueeParticle,
)

from .control_panel import (
    ControlPanel,
    LabeledSlider,
    LabeledSpinBox,
    TyphoonCategorySelector,
)

from .dashboard import (
    Dashboard,
    MetricCard,
    ProgressMetricCard,
    ShelterStatusCard,
    TimeEstimateCard,
    RouteStatusCard,
)

from .comparison_view import (
    ComparisonView,
    ConvergenceChart,
    PerformanceTable,
    RadarChart,
    WinnerBadge,
)

from .main_window import (
    MainWindow,
    OptimizationWorker,
    SimulationWorker,
    run_app,
)

__all__ = [
    # Styles
    'COLORS',
    'ColorPalette',
    'MAIN_STYLESHEET',
    'Sizes',
    'Animation',
    'MapStyle',
    'hex_to_rgb',
    'rgba_from_hex',

    # Map Widget
    'MapWidget',
    'MapCanvas',
    'PopulationZoneItem',
    'ShelterItem',
    'HazardZoneItem',
    'RouteItem',
    'EvacueeParticle',

    # Control Panel
    'ControlPanel',
    'LabeledSlider',
    'LabeledSpinBox',
    'TyphoonCategorySelector',

    # Dashboard
    'Dashboard',
    'MetricCard',
    'ProgressMetricCard',
    'ShelterStatusCard',
    'TimeEstimateCard',
    'RouteStatusCard',

    # Comparison View
    'ComparisonView',
    'ConvergenceChart',
    'PerformanceTable',
    'RadarChart',
    'WinnerBadge',

    # Main Window
    'MainWindow',
    'OptimizationWorker',
    'SimulationWorker',
    'run_app',
]
