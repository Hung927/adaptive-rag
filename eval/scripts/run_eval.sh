#!/bin/bash
# RAG evaluation runner
set -e

MODE="${1:---both}"

echo "=== RAG Evaluation ==="
echo "Mode: $MODE"
echo ""

case "$MODE" in
    --retrieval)
        uv run pytest eval/tests/test_retrieval.py --override-ini="testpaths=eval/tests" -v
        ;;
    --generation)
        uv run pytest eval/tests/test_generation.py --override-ini="testpaths=eval/tests" -v
        ;;
    --both)
        uv run pytest eval/tests/test_retrieval.py eval/tests/test_generation.py --override-ini="testpaths=eval/tests" -v
        ;;
    --comparison)
        # Run all 4 review combinations (A/B/C/D) and output comparison CSV
        uv run pytest eval/tests/test_review_comparison.py --override-ini="testpaths=eval/tests" -v -s
        ;;
    --all)
        uv run pytest eval/tests/ --override-ini="testpaths=eval/tests" -v -s
        ;;
    *)
        echo "Usage: $0 [--retrieval|--generation|--both|--comparison|--all]"
        exit 1
        ;;
esac
