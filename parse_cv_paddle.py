"""Parse a PDF with PaddleOCR's non-VLM document-structure pipeline.

The output is a single Markdown document plus per-page JSON/Markdown assets.
PP-StructureV3 combines text detection/recognition, layout detection and
reading-order reconstruction; it is deliberately separate from test.py,
which exercises PaddleOCR-VL.

This script is configured for PP-DocLayout_plus-L, PP-OCRv6_tiny_det, and
PP-OCRv6_tiny_rec models. ONNX
Runtime is used for the complete PP-StructureV3 pipeline by default.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def markdown_text(value: object) -> str:
    """Extract text from PaddleX's versioned MarkdownResult wrapper."""
    if isinstance(value, str):
        return value
    text = getattr(value, "markdown_texts", None)
    if text is None and isinstance(value, dict):
        text = value.get("markdown_texts") or value.get("markdown_text")
    if isinstance(text, str):
        return text
    if isinstance(text, (list, tuple)) and all(isinstance(item, str) for item in text):
        return "\n\n".join(text)
    raise TypeError(
        f"Unsupported PaddleOCR Markdown result: {type(value).__name__}; "
        "expected a string or an object with markdown_texts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="PDF or image to parse",
        default=Path("data/Pham Hong Trang_CV_Product & Channel MKT.pdf"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/paddle_structure"),
        help="Directory for Markdown, JSON, and extracted images",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Paddle device, e.g. cpu or gpu:0 (default: cpu)",
    )
    parser.add_argument(
        "--doc-orientation",
        action="store_true",
        help="Enable document orientation classification",
    )
    parser.add_argument(
        "--unwarp",
        action="store_true",
        help="Enable document unwarping",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Optional root containing local model directories; otherwise models are downloaded",
    )
    parser.add_argument(
        "--layout-model-dir",
        type=Path,
        default=None,
        help="Local PP-DocLayout_plus-L directory (overrides --model-root)",
    )
    parser.add_argument(
        "--layout-model-name",
        default="PP-DocLayout_plus-L",
        help="Layout model to download when --layout-model-dir is omitted",
    )
    parser.add_argument(
        "--det-model-dir",
        type=Path,
        default=None,
        help="Local PP-OCRv6_tiny_det directory (overrides --model-root)",
    )
    parser.add_argument(
        "--rec-model-dir",
        type=Path,
        default=None,
        help="Local PP-OCRv6_tiny_rec directory (overrides --model-root)",
    )
    parser.add_argument(
        "--det-model-name",
        default="PP-OCRv6_tiny_det",
        help="Model name to auto-download when --det-model-dir is omitted",
    )
    parser.add_argument(
        "--rec-model-name",
        default="PP-OCRv6_tiny_rec",
        help="Model name to auto-download when --rec-model-dir is omitted",
    )
    parser.add_argument(
        "--engine",
        choices=("onnxruntime", "paddle", "paddle_static", "paddle_dynamic"),
        default="onnxruntime",
        help="Inference engine (default: onnxruntime)",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table recognition modules",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input does not exist: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_root = args.model_root
    layout_model_dir = args.layout_model_dir or (
        model_root / "PP-DocLayout_plus-L" if model_root else None
    )
    det_model_dir = args.det_model_dir or (
        model_root / "PP-OCRv6_tiny_det" if model_root else None
    )
    rec_model_dir = args.rec_model_dir or (
        model_root / "PP-OCRv6_tiny_rec" if model_root else None
    )
    model_dirs = {
        "PP-DocLayout_plus-L": layout_model_dir,
        "PP-OCRv6_tiny_det": det_model_dir,
        "PP-OCRv6_tiny_rec": rec_model_dir,
    }
    missing = [
        f"{name}: {path}"
        for name, path in model_dirs.items()
        if path is not None and not path.is_dir()
    ]
    if missing:
        raise SystemExit(
            "Missing local Paddle model directories:\n  " + "\n  ".join(missing)
        )

    from paddleocr import PPStructureV3
    from paddlex.inference import load_pipeline_config

    pipeline_started_at = time.perf_counter()
    # A local directory is passed without model_name so PaddleX never tries to
    # resolve that component through the remote model registry. If no local
    # directory is supplied, model_name enables PaddleX's normal auto-download.
    model_args = {
        "layout_detection_model_name": (
            None if layout_model_dir else args.layout_model_name
        ),
        "layout_detection_model_dir": (
            str(layout_model_dir) if layout_model_dir else None
        ),
        "text_detection_model_name": None if det_model_dir else args.det_model_name,
        "text_detection_model_dir": str(det_model_dir) if det_model_dir else None,
        "text_recognition_model_name": None if rec_model_dir else args.rec_model_name,
        "text_recognition_model_dir": str(rec_model_dir) if rec_model_dir else None,
    }
    # PPStructureV3 3.7.0 does not expose `engine` in its Python signature.
    # Configure PaddleX directly instead. PP-DocLayout_plus-L provides the
    # ONNX package required for full-pipeline ONNX inference.
    paddlex_config = load_pipeline_config("PP-StructureV3")
    paddlex_config["engine"] = args.engine

    pipeline = PPStructureV3(
        paddlex_config=paddlex_config,
        device=args.device,
        **model_args,
        use_doc_orientation_classify=args.doc_orientation,
        use_doc_unwarping=args.unwarp,
        use_textline_orientation=False,
        use_region_detection=False,
        use_table_recognition=not args.no_tables,
        wired_table_structure_recognition_model_name="SLANet",
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )
    pipeline_init_seconds = time.perf_counter() - pipeline_started_at
    print(f"Pipeline initialized in {pipeline_init_seconds:.2f}s")

    page_markdown = []
    inference_started_at = time.perf_counter()
    results = iter(pipeline.predict(input=str(args.input)))
    print(
        f"Page completed in {time.perf_counter() - inference_started_at:.2f}s"
    )
    page_number = 0
    while True:
        page_started_at = time.perf_counter()
        try:
            result = next(results)
        except StopIteration:
            break
        page_number += 1
        # result.print()
        # result.save_to_json(save_path=str(args.output_dir))
        result.save_to_markdown(save_path=str(args.output_dir))
        page_markdown.append(result.markdown)
        print(
            f"Page {page_number} completed in "
            f"{time.perf_counter() - page_started_at:.2f}s"
        )

    if not page_markdown:
        raise SystemExit("PaddleOCR returned no pages")

    markdown = markdown_text(pipeline.concatenate_markdown_pages(page_markdown))
    output_file = args.output_dir / f"{args.input.stem}.md"
    output_file.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output_file}")
    print(f"Inference completed in {time.perf_counter() - inference_started_at:.2f}s")
    print(
        f"Total runtime: {time.perf_counter() - pipeline_started_at:.2f}s "
        f"(including pipeline initialization)"
    )


if __name__ == "__main__":
    main()
