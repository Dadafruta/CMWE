.PHONY: help lfs-check py-check daily-v4 eval-latest tree

help:
	@echo "Common targets:"
	@echo "  make lfs-check   - verify Git LFS pointers/files"
	@echo "  make py-check    - compile Python files (syntax/import sanity)"
	@echo "  make daily-v4    - run full daily_mathgate_v4 pipeline"
	@echo "  make eval-latest - re-eval latest daily_bench_v4 run"
	@echo "  make tree        - show top-level repo structure"

lfs-check:
	git lfs install
	git lfs fsck
	git lfs ls-files | head -n 20

py-check:
	python -m compileall -q .

daily-v4:
	bash scripts/run_daily_mathgate_v4_pipeline.sh

eval-latest:
	bash scripts/eval_latest_daily_mathgate_v4.sh

tree:
	@find . -maxdepth 2 -type d \
	  -not -path './.git*' \
	  -not -path './.venv*' \
	  -print | sed 's|^\./||' | sort
