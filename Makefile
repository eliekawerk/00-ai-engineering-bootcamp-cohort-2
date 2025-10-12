run-docker-compose:
	uv sync
	docker compose up --build -d

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

stop-docker-compose:
	docker compose down	