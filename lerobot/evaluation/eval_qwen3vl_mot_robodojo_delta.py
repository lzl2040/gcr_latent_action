"""Evaluate delta-trained RoboDojo stack-bowls policies using X-WAM-style integration."""

from pathlib import Path

from lerobot.evaluation.eval_qwen3vl_mot_robodojo import build_parser as base_parser
from lerobot.evaluation.eval_qwen3vl_mot_robodojo import main as run

EXPERIMENT = Path("/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu")
DATASET = Path("/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/stack_bowls_delta")


def build_parser():
    parser = base_parser()
    parser.description = __doc__
    parser.set_defaults(
        action_mode="delta",
        checkpoint=EXPERIMENT / "step8k",
        dataset_root=DATASET,
        diagnostic_episodes=[0, 1, 2],
        timeout=7200,
    )
    return parser


def main() -> None:
    run(build_parser())


if __name__ == "__main__":
    main()
