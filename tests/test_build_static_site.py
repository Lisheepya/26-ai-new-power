import json
import tempfile
import unittest
from pathlib import Path

from build_static_site import build_static_site


class BuildStaticSiteTests(unittest.TestCase):
    temp_parent = Path(__file__).resolve().parents[1]

    def test_repository_result_values_are_copied_verbatim_from_existing_json(self):
        root = self.temp_parent
        build_static_site(root)
        site_results = json.loads((root / "docs" / "results.json").read_text(encoding="utf-8"))
        expected_sources = {
            "iris": ("iris_metrics", "outputs/metrics.json"),
            "ai4i": ("ai4i_metrics", "outputs/ai4i_modeling/model_comparison.json"),
            "ai4i_eda": ("ai4i_eda", "outputs/ai4i_eda/summary.json"),
        }
        for result_key, (source_key, source_name) in expected_sources.items():
            expected = json.loads((root / source_name).read_text(encoding="utf-8"))
            self.assertEqual(site_results[result_key], expected)
            self.assertEqual(site_results["source_files"][source_key], source_name)

        page = (root / "docs" / "index.html").read_text(encoding="utf-8")
        app = (root / "docs" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="results-container"', page)
        self.assertIn('root.fetch("results.json")', app)
        self.assertIn('id="iris-form"', page)
        self.assertIn('id="ai4i-form"', page)

    def make_project(self, root: Path) -> dict[str, Path]:
        outputs = root / "outputs"
        (outputs / "ai4i").mkdir(parents=True)
        (outputs / "ai4i_modeling").mkdir()
        (outputs / "ai4i_eda").mkdir()
        (outputs / "小論文.md").write_text(
            "# Iris report\n\nAccuracy: **0.75**\n\n| Metric | Value |\n|---|---:|\n| accuracy | 0.75 |\n\n![Iris plot](iris_plot.png)\n",
            encoding="utf-8",
        )
        (outputs / "AI4I_小論文.md").write_text(
            "# AI4I report\n\nSee [metrics](ai4i_modeling/model_comparison.json).\n\n![AI4I plot](ai4i/plot.png)\n",
            encoding="utf-8",
        )
        (outputs / "metrics.json").write_text(
            json.dumps({"accuracy": 0.75, "test_samples": 2}), encoding="utf-8"
        )
        (outputs / "ai4i_modeling" / "model_comparison.json").write_text(
            json.dumps({"models": {"mlp": {"metrics": {"recall": 0.5}}}}),
            encoding="utf-8",
        )
        (outputs / "ai4i_eda" / "summary.json").write_text(
            json.dumps({"rows": 4}), encoding="utf-8"
        )
        iris_figure = outputs / "iris_plot.png"
        ai4i_figure = outputs / "ai4i" / "plot.png"
        iris_figure.write_bytes(b"original iris figure bytes")
        ai4i_figure.write_bytes(b"original ai4i figure bytes")
        (root / "docs").mkdir()
        (root / "docs" / "index.html").write_text("preserve homepage", encoding="utf-8")
        return {"iris_figure": iris_figure, "ai4i_figure": ai4i_figure}

    def test_copies_existing_numbers_figures_and_html_reports_without_overwriting_home(self):
        with tempfile.TemporaryDirectory(dir=self.temp_parent) as temp_dir:
            root = Path(temp_dir)
            figures = self.make_project(root)

            result = build_static_site(root)

            docs = root / "docs"
            copied = json.loads((docs / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(copied["iris"], {"accuracy": 0.75, "test_samples": 2})
            self.assertEqual(
                copied["ai4i"]["models"]["mlp"]["metrics"]["recall"], 0.5
            )
            self.assertEqual((docs / "figures" / "iris" / "iris_plot.png").read_bytes(), figures["iris_figure"].read_bytes())
            self.assertEqual((docs / "figures" / "ai4i" / "plot.png").read_bytes(), figures["ai4i_figure"].read_bytes())
            iris_html = (docs / "report_iris.html").read_text(encoding="utf-8")
            self.assertIn("<h1>Iris report</h1>", iris_html)
            self.assertIn("<table>", iris_html)
            self.assertIn('src="figures/iris/iris_plot.png"', iris_html)
            ai4i_html = (docs / "report_ai4i.html").read_text(encoding="utf-8")
            self.assertIn('href="results.json"', ai4i_html)
            self.assertIn('src="figures/ai4i/plot.png"', ai4i_html)
            self.assertEqual((docs / "index.html").read_text(encoding="utf-8"), "preserve homepage")
            self.assertEqual(result["figure_count"], 2)

    def test_requires_existing_evaluation_artifacts_before_writing(self):
        with tempfile.TemporaryDirectory(dir=self.temp_parent) as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                build_static_site(root)
            self.assertFalse((root / "docs").exists())


if __name__ == "__main__":
    unittest.main()
