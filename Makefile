PYTHON ?= python
SPLIT ?= dev

NER_PRIMARY_CACHE := models/ner_bert_ds
NER_SECONDARY_CACHE := models/ner_distil_supervised
RE_CACHE := models/re_distant.pkl

NER_PRIMARY_ARGS := --distant-train data/distant_10000.json --pretrained-model bert-base-cased --distant-epochs 1 --distant-learning-rate 3e-5 --epochs 2 --learning-rate 3e-5 --batch-size 32 --outside-weight 0.6 --cache $(NER_PRIMARY_CACHE)
NER_SECONDARY_ARGS := --pretrained-model distilbert-base-cased --epochs 2 --learning-rate 5e-5 --batch-size 32 --outside-weight 0.6 --cache $(NER_SECONDARY_CACHE)
RE_ARGS := --distant-train data/distant_10000.json --distant-weight 0.25 --max-relations 2 --relation-no-ratio 0.90 --cache $(RE_CACHE)

.PHONY: dirs train-ner predict-ner eval-ner train-re predict-re eval-re validate

dirs:
	mkdir -p models artifacts predictions results

train-ner: dirs
	$(PYTHON) ner_predict.py data/dev_unlabeled.json artifacts/ner_primary_dev.json $(NER_PRIMARY_ARGS)
	$(PYTHON) ner_predict.py data/dev_unlabeled.json artifacts/ner_secondary_dev.json $(NER_SECONDARY_ARGS)
	$(PYTHON) ner_ensemble.py artifacts/ner_primary_dev.json artifacts/ner_secondary_dev.json predictions/ner_dev.json

predict-ner: dirs
	$(PYTHON) ner_predict.py data/$(SPLIT)_unlabeled.json artifacts/ner_primary_$(SPLIT).json $(NER_PRIMARY_ARGS)
	$(PYTHON) ner_predict.py data/$(SPLIT)_unlabeled.json artifacts/ner_secondary_$(SPLIT).json $(NER_SECONDARY_ARGS)
	$(PYTHON) ner_ensemble.py artifacts/ner_primary_$(SPLIT).json artifacts/ner_secondary_$(SPLIT).json predictions/ner_$(SPLIT).json

eval-ner:
	$(PYTHON) ner_eval.py data/$(SPLIT)_revised.json predictions/ner_$(SPLIT).json --json results/ner_$(SPLIT)_scores.json

train-re: dirs
	$(PYTHON) re_predict.py predictions/ner_dev.json artifacts/re_dev_traincheck.json $(RE_ARGS)

predict-re: dirs
	$(PYTHON) re_predict.py predictions/ner_$(SPLIT).json predictions/re_$(SPLIT).json $(RE_ARGS)

eval-re:
	$(PYTHON) re_eval.py data/$(SPLIT)_revised.json predictions/re_$(SPLIT).json --names data/rel_info.json --json results/re_$(SPLIT)_scores.json

validate:
	$(PYTHON) check_format.py data/$(SPLIT)_unlabeled.json predictions/ner_$(SPLIT).json
	$(PYTHON) check_format.py --re --names data/rel_info.json predictions/ner_$(SPLIT).json predictions/re_$(SPLIT).json
