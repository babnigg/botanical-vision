# Developer entrypoints for the Botanical Vision MLOps flow.
# Windows without `make`: run the commands after each target directly.
PY  ?= python
LIMIT ?= 300

.PHONY: help mlflow-ui register-baseline register evaluate promote flow serve test lint docker-build docker-run

help:
	@echo "mlflow-ui | register-baseline | register CKPT=... | evaluate V=.. | promote V=.. | flow CKPT=.. | serve | test | lint | docker-build | docker-run"

mlflow-ui:            ## browse the registry + runs
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db --port 5000

register-baseline:    ## backfill baseline as @production
	$(PY) -m mlops.register_checkpoint --checkpoint checkpoints/resnet50_baseline.pt --alias production

register:             ## register any checkpoint: make register CKPT=checkpoints/x.pt
	$(PY) -m mlops.register_checkpoint --checkpoint $(CKPT)

evaluate:             ## make evaluate V=2 LIMIT=300
	$(PY) -m mlops.evaluate --version $(V) --limit $(LIMIT)

promote:              ## champion/challenger gate: make promote V=2
	$(PY) -m mlops.promote --challenger $(V) --margin 0.005 --limit $(LIMIT)

flow:                 ## Prefect train->register->gate: make flow CKPT=checkpoints/x.pt
	$(PY) -m mlops.flows.train_flow --checkpoint $(CKPT) --limit $(LIMIT)

serve:                ## run the API (dev)
	$(PY) -m uvicorn api.main:app --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check api mlops

docker-build:
	docker build -f api/Dockerfile -t botanical-vision-api .

docker-run:           ## mounts the local registry so the container can serve the model
	docker run --rm -p 8000:8000 \
	  -v "$(PWD)/mlops:/app/mlops" -v "$(PWD)/mlruns:/app/mlruns" \
	  botanical-vision-api
