docker-build:
	docker build . -t alzheimers-classification
run:
	docker run --rm  -it alzheimers-classification bash
run-gpu:
	docker run --rm --gpus all -it alzheimers-classification bash
copy-results:
	@read -p "Container id:" contId \
	&& docker cp $$contId:/AlzheimersClassification/results .