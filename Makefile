run-docker-compose:
	uv sync
	docker compose up --build -d

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

stop-docker-compose:
	docker compose down	

run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever

prune-docker:
	docker system prune -af  	