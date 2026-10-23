"""TDD contract for the reproducible AI4I paper and README workflow."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai4i_report import DEFAULT_OUTPUT_PATH, _split_descriptions, generate_ai4i_paper


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_FIGURES = (
    "pca_3d.png",
    "confusion_matrices.png",
    "precision_recall_curve.png",
    "roc_curve.png",
    "mlp_loss_curve.png",
    "mlp_permutation_importance.png",
)
REQUIRED_SECTIONS = (
    "摘要",
    "研究背景",
    "資料集與資料探索",
    "研究方法",
    "實驗結果",
    "討論",
    "研究限制",
    "結論",
    "參考資料",
)


class AI4IReportTests(unittest.TestCase):
    def test_generates_paper_linked_to_actual_metrics_and_figures(self):
        eda_path = ROOT_DIR / "outputs" / "ai4i_eda" / "summary.json"
        model_path = ROOT_DIR / "outputs" / "ai4i_modeling" / "model_comparison.json"
        eda = json.loads(eda_path.read_text(encoding="utf-8"))
        models = json.loads(model_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory(dir=DEFAULT_OUTPUT_PATH.parent) as temp_dir:
            output_path = Path(temp_dir) / DEFAULT_OUTPUT_PATH.name
            generated_path = generate_ai4i_paper(
                eda_summary_path=eda_path,
                model_results_path=model_path,
                output_path=output_path,
            )

            self.assertEqual(generated_path, output_path)
            self.assertTrue(output_path.is_file())
            content = output_path.read_text(encoding="utf-8")

            figure_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
            for filename in EXPECTED_FIGURES:
                with self.subTest(figure=filename):
                    relative_link = next(
                        (link for link in figure_links if link.endswith(filename)),
                        None,
                    )
                    self.assertIsNotNone(relative_link, f"missing figure link: {filename}")
                    self.assertTrue(
                        (output_path.parent / relative_link).is_file(),
                        f"broken figure: {relative_link}",
                    )
            local_links = [
                link
                for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content)
                if not link.startswith(("https://", "http://"))
            ]
            for link in local_links:
                with self.subTest(local_link=link):
                    self.assertTrue((output_path.parent / link).is_file(), f"broken link: {link}")

        headings = [
            re.sub(r"^\d+\.\s*", "", heading)
            for heading in re.findall(r"^## (.+)$", content, flags=re.MULTILINE)
        ]
        for section in REQUIRED_SECTIONS:
            with self.subTest(section=section):
                self.assertIn(section, headings)

        self.assertIn("synthetic", content.casefold())
        self.assertIn("合成資料", content)
        self.assertIn("10.24432/C5HS5C", content)
        self.assertIn("Machine failure", content)
        self.assertIn("隨機切分", content)
        self.assertIn("不代表真實產線", content)
        self.assertIn("average precision", content.casefold())
        self.assertIn("事後解釋", content)
        self.assertIn("同一筆觀測", content)
        self.assertIn("提前預警", content)
        self.assertIn(f"{eda['rows']:,}", content)
        self.assertIn(str(eda["column_count"]), content)
        self.assertIn(eda["data_sha256"], content)
        self.assertIn("ai4i_eda/summary.json", content)
        self.assertIn("ai4i_modeling/model_comparison.json", content)
        self.assertIn(".venv\\Scripts\\python.exe ai4i_report.py", content)
        self.assertIn("python -m venv .venv", content)

        for model_key in ("logistic_regression", "mlp"):
            model_metrics = models["models"][model_key]["metrics"]
            for metric_name in ("accuracy", "precision", "recall", "f1"):
                self.assertIn(f"{model_metrics[metric_name]:.2%}", content)
            for metric_name in ("roc_auc", "pr_auc"):
                self.assertIn(f"{model_metrics[metric_name]:.3f}", content)
            matrix = model_metrics["confusion_matrix"]
            true_negative, false_positive = matrix[0]
            false_negative, true_positive = matrix[1]
            label = "MLP" if model_key == "mlp" else "Logistic Regression"
            self.assertIn(
                f"| {label} | {true_negative} | {false_positive} | "
                f"{false_negative} | {true_positive} |",
                content,
            )

        split = models["metadata"]["split"]
        self.assertIn(f"{split['train_samples']:,}", content)
        self.assertIn(f"{split['test_samples']:,}", content)
        self.assertIn(str(models["metadata"]["random_state"]), content)
        self.assertIn(str(split["train_class_counts"]["0"]), content)
        self.assertIn(str(split["train_class_counts"]["1"]), content)
        self.assertIn(str(split["test_class_counts"]["0"]), content)
        self.assertIn(str(split["test_class_counts"]["1"]), content)
        self.assertIn(str(eda["missing_values"]), content)
        self.assertIn(str(eda["duplicate_rows"]), content)
        self.assertIn(f"{eda['label_prevalence']['Machine failure']['positive_prevalence_percent']:.2f}%", content)
        for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
            self.assertIn(f"{mode}={eda['label_counts'][mode]['1']}", content)
        correlation = eda["sensor_correlation_matrix"]
        self.assertIn(
            f"{correlation['Air temperature [K]']['Process temperature [K]']:.3f}",
            content,
        )
        self.assertIn(
            f"{correlation['Rotational speed [rpm]']['Torque [Nm]']:.3f}",
            content,
        )
        importance = models["models"]["mlp"]["permutation_importance"]
        for value in importance["importances_mean"] + importance["importances_std"]:
            self.assertIn(f"{value:.4f}", content)
        parameters = models["models"]["mlp"]["model_params"]
        self.assertIn(", ".join(map(str, parameters["hidden_layer_sizes"])), content)
        self.assertIn(parameters["activation"], content)
        self.assertIn(parameters["solver"], content)
        self.assertIn(f"{parameters['learning_rate_init']:g}", content)
        self.assertIn(f"{parameters['alpha']:g}", content)
        self.assertIn(str(parameters["max_iter"]), content)
        self.assertIn(str(eda["label_counts"]["Machine failure"]["1"]), content)

    def test_rejects_mismatched_source_fingerprint_and_missing_figures(self):
        eda_path = ROOT_DIR / "outputs" / "ai4i_eda" / "summary.json"
        model_path = ROOT_DIR / "outputs" / "ai4i_modeling" / "model_comparison.json"
        eda = json.loads(eda_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory(dir=DEFAULT_OUTPUT_PATH.parent) as temp_dir:
            root = Path(temp_dir)
            mismatched_eda_path = root / "mismatched_summary.json"
            eda["data_sha256"] = "0" * 64
            mismatched_eda_path.write_text(json.dumps(eda), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                generate_ai4i_paper(
                    eda_summary_path=mismatched_eda_path,
                    model_results_path=model_path,
                    output_path=root / "mismatched.md",
                )

            with patch("ai4i_report.FIGURE_DIR", root / "missing_figures"):
                with self.assertRaisesRegex(FileNotFoundError, "figure"):
                    generate_ai4i_paper(
                        eda_summary_path=eda_path,
                        model_results_path=model_path,
                        output_path=root / "no_figures.md",
                    )

    def test_udi_ordered_split_description_is_sensitivity_not_temporal(self):
        summary, description = _split_descriptions(
            {"strategy": "udi_ordered", "stratified": False},
            rows=10_000,
            train_samples=8_000,
            test_samples=2_000,
            test_size=0.2,
            random_state=42,
        )

        self.assertIn("UDI 順序列區塊敏感度切分", summary)
        self.assertIn("不是時間驗證", summary)
        self.assertIn("前 80% 作訓練、後 20% 作測試", description)
        self.assertNotIn("分層切分", description)

    def test_readme_documents_clean_environment_reproduction(self):
        readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

        workflow = readme.split("## AI4I 小論文與完整重現流程", 1)[1].split(
            "## 資料來源", 1
        )[0]
        commands = (
            "py -m venv .venv",
            "-m pip install -r requirements.txt",
            "ai4i_data.py --download",
            "ai4i_eda.py",
            "ai4i_modeling.py --test-size 0.2 --random-state 42",
            "ai4i_visualization.py",
            "ai4i_report.py",
            "-m unittest discover -s tests -v",
        )
        positions = [workflow.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("python -m venv .venv", workflow)
        self.assertIn("outputs/AI4I_小論文.md", readme)
        self.assertIn("outputs/小論文.md", readme)
        self.assertIn("--split-strategy udi_ordered", readme)
        self.assertIn("UDI 順序不等於已知時間，這不是時間驗證", readme)


if __name__ == "__main__":
    unittest.main()
