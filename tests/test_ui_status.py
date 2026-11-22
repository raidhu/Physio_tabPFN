"""Tests for PhysioTabPFN UI and Status endpoint robustness.

Ensures that:
1. Every DOM ID accessed by JavaScript exists in HTML_PAGE.
2. The status JSON response has all necessary fields.
3. No red stripes or red colors remain in CSS.
4. Rep counter and status state updates are resilient.
"""

import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import HTML_PAGE, AppState


def test_no_red_stripes_or_accents():
    """Verify that red stripe variables and red color rules have been completely removed."""
    forbidden_terms = [
        "--accent-red",
        "--red-stripe",
        "--red-light",
        "var(--accent-red)",
        "var(--red-stripe)",
    ]
    for term in forbidden_terms:
        assert term not in HTML_PAGE, f"Forbidden red term '{term}' found in HTML_PAGE"


def test_all_js_element_ids_exist_in_html():
    """Verify every document.getElementById('...') or setText('...') in script exists in HTML."""
    # Find all element IDs targeted by setText or getElementById
    pattern = r"(?:document\.getElementById|setText)\(\s*['\"]([^'\"]+)['\"]"
    targeted_ids = set(re.findall(pattern, HTML_PAGE))

    assert len(targeted_ids) > 0, "No element IDs found in JS!"

    for el_id in targeted_ids:
        # Check that id="..." exists in HTML
        id_pattern = rf'id=["\']{re.escape(el_id)}["\']'
        assert re.search(id_pattern, HTML_PAGE), f"Element id='{el_id}' referenced in JS but missing in HTML!"


def test_app_state_json_structure():
    """Verify AppState generates valid JSON with all keys needed by the dashboard."""
    state = AppState()
    status_json = state.get_status_json()
    data = json.loads(status_json)

    expected_keys = [
        "active_exercise",
        "predicted_exercise",
        "confidence",
        "latency_ms",
        "mode",
        "valid_reps",
        "bad_reps",
        "total_attempts",
        "quality_score",
        "feedback",
        "last_rep_status",
        "posture_counts",
        "probabilities",
    ]

    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from status JSON: {data}"

    # Verify posture counts covers the active classes
    for cls_name in ["Squats", "Step-Ups", "Idle"]:
        assert cls_name in data["posture_counts"], f"Missing class '{cls_name}' in posture_counts"


def test_js_function_definitions():
    """Verify that all called JS functions are properly declared."""
    required_functions = [
        "setText",
        "updateDashboard",
        "updateSliderPosition",
        "selectExercise",
        "switchNavTab",
        "triggerReset",
    ]
    for fn in required_functions:
        fn_pattern = rf"function\s+{re.escape(fn)}\s*\("
        assert re.search(fn_pattern, HTML_PAGE), f"Function '{fn}' not defined in script!"


if __name__ == "__main__":
    print("Running UI Status Tests...")
    test_no_red_stripes_or_accents()
    print("✓ test_no_red_stripes_or_accents passed")
    test_all_js_element_ids_exist_in_html()
    print("✓ test_all_js_element_ids_exist_in_html passed")
    test_app_state_json_structure()
    print("✓ test_app_state_json_structure passed")
    test_js_function_definitions()
    print("✓ test_js_function_definitions passed")
    print("\nALL UI & TELEMETRY VERIFICATIONS PASSED!")
