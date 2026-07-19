# The whole team job is two commands: share a model, compare everyone's.
PY    ?= python
LIMIT ?= 300

.PHONY: help publish leaderboard score lint

help:
	@echo "publish CKPT=checkpoints/x.pt NAME=my-model  - share your model with the team"
	@echo "leaderboard                                  - rank everyone's on the same test split"
	@echo "score CKPT=checkpoints/x.pt                  - check your own top-1/top-5 first"
	@echo "lint                                         - ruff check share/ scripts/"

publish:            ## share your model: make publish CKPT=checkpoints/x.pt NAME=my-model
	$(PY) -m share.publish --checkpoint $(CKPT) --name $(NAME)

leaderboard:        ## rank everyone's shared models on the same test split
	$(PY) -m share.leaderboard --limit $(LIMIT)

score:              ## check your own checkpoint before publishing
	$(PY) -m share.score --checkpoint $(CKPT) --limit $(LIMIT)

lint:
	$(PY) -m ruff check share scripts
