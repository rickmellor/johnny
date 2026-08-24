"""PlanBench suite: action normalization, scoring, and wiring."""
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "planbench_eval", Path(__file__).parent.parent / "src/johnny/scripts/planbench_eval.py")
pb = importlib.util.module_from_spec(spec); sys.modules["planbench_eval"] = pb; spec.loader.exec_module(pb)


def test_pddl_reference_parses_to_actions():
    assert pb.actions_from_pddl("(pick-up red)\n(stack red orange)") == [
        ("pick-up", "red"), ("stack", "red", "orange")]
    assert pb.actions_from_pddl("") == []


def test_natural_language_plan_normalizes_to_same_actions():
    text = ("[PLAN]\npick up the red block\nstack the red block on top of the orange block\n[PLAN END]")
    assert pb.actions_from_text(text) == [("pick-up", "red"), ("stack", "red", "orange")]


def test_unstack_and_putdown_forms():
    text = "unstack the blue block from on top of the red block\nput down the blue block"
    assert pb.actions_from_text(text) == [("unstack", "blue", "red"), ("put-down", "blue")]


def test_prose_around_the_plan_is_ignored():
    text = ("Sure! Here is my plan.\n[PLAN]\npick up the red block\n[PLAN END]\n"
            "Let me know if you want another approach.")
    assert pb.actions_from_text(text) == [("pick-up", "red")]


def test_prefix_match_partial_credit():
    gt = [("pick-up", "red"), ("stack", "red", "orange"), ("pick-up", "blue")]
    assert pb.prefix_match(gt, gt) == 3
    assert pb.prefix_match([("pick-up", "red"), ("stack", "red", "blue")], gt) == 1
    assert pb.prefix_match([], gt) == 0


def test_suite_is_registered_and_dispatched():
    from johnny import bench
    assert "planbench" in bench.SUITES
    assert hasattr(bench, "_run_planbench")
