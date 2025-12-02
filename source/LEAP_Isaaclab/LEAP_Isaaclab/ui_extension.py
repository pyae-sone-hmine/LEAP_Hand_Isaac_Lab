# filepath: /home/pyae/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/ui_extension.py
# --------------------------------------------------------
# LEAP Hand Goal Angle Control UI Extension
# --------------------------------------------------------

import math
import weakref

# Global reference to the environment (set by play.py)
_env_ref = None


def set_env_reference(env):
    """Set a weak reference to the environment for UI control."""
    global _env_ref
    _env_ref = weakref.ref(env)
    print("[LEAP UI] Environment reference set")


def get_env():
    """Get the environment reference."""
    global _env_ref
    if _env_ref is not None:
        return _env_ref()
    return None


class LeapHandGoalAngleUI:
    """UI Window for controlling the goal angle of the LEAP Hand reorientation task."""
    
    def __init__(self):
        self._window = None
        self._slider = None
        self._label = None
        self._current_angle_deg = 0.0
        
    def build_ui(self):
        """Build the UI window with slider control."""
        try:
            import omni.ui as ui
        except ImportError:
            print("[LEAP UI] omni.ui not available - UI disabled")
            return
            
        self._window = ui.Window("LEAP Hand Goal Control", width=350, height=200)
        
        with self._window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Goal Angle Control", height=30, style={"font_size": 18})
                
                with ui.HStack(height=30):
                    ui.Label("Angle (degrees):", width=120)
                    self._label = ui.Label(f"{self._current_angle_deg:.1f}°", width=80)
                
                # Slider from -180 to 180 degrees
                self._slider = ui.FloatSlider(min=-180.0, max=180.0, step=1.0)
                self._slider.model.set_value(self._current_angle_deg)
                self._slider.model.add_value_changed_fn(self._on_slider_changed)
                
                # with ui.HStack(height=30, spacing=10):
                #     ui.Button("0°", clicked_fn=lambda: self._set_angle(0.0), width=50)
                #     ui.Button("45°", clicked_fn=lambda: self._set_angle(45.0), width=50)
                #     ui.Button("90°", clicked_fn=lambda: self._set_angle(90.0), width=50)
                #     ui.Button("180°", clicked_fn=lambda: self._set_angle(180.0), width=50)
                #     ui.Button("-90°", clicked_fn=lambda: self._set_angle(-90.0), width=50)
                
                ui.Spacer(height=10)
                
                with ui.HStack(height=25):
                    ui.Label("Status:", width=60)
                    self._status_label = ui.Label("Ready", style={"color": 0xFF00FF00})
    
    def _on_slider_changed(self, model):
        """Handle slider value changes."""
        angle_deg = model.get_value_as_float()
        self._set_angle(angle_deg, update_slider=False)
    
    def _set_angle(self, angle_deg: float, update_slider: bool = True):
        """Set the goal angle."""
        self._current_angle_deg = angle_deg
        
        if self._label is not None:
            self._label.text = f"{angle_deg:.1f}°"
        
        if update_slider and self._slider is not None:
            self._slider.model.set_value(angle_deg)
        
        # Update the environment
        env = get_env()
        if env is not None:
            try:
                env.set_goal_angle_deg(angle_deg)
                if hasattr(self, '_status_label') and self._status_label is not None:
                    self._status_label.text = f"Set to {angle_deg:.1f}°"
            except Exception as e:
                print(f"[LEAP UI] Error setting angle: {e}")
                if hasattr(self, '_status_label') and self._status_label is not None:
                    self._status_label.text = f"Error: {e}"
        else:
            if hasattr(self, '_status_label') and self._status_label is not None:
                self._status_label.text = "No environment connected"
    
    def destroy(self):
        """Clean up the UI."""
        if self._window is not None:
            self._window.destroy()
            self._window = None


# Global UI instance
_goal_angle_ui = None


def create_goal_angle_ui():
    """Create and show the goal angle UI."""
    global _goal_angle_ui
    if _goal_angle_ui is None:
        _goal_angle_ui = LeapHandGoalAngleUI()
    _goal_angle_ui.build_ui()
    return _goal_angle_ui


def destroy_goal_angle_ui():
    """Destroy the goal angle UI."""
    global _goal_angle_ui
    if _goal_angle_ui is not None:
        _goal_angle_ui.destroy()
        _goal_angle_ui = None
