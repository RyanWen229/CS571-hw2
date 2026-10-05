PYTHON ?= python
SPLIT ?= dev

NER_CACHE := models/ner_feature_ds.pkl
RE_CACHE := models/re_distant.pkl

NER_ARGS := --distant-train data/distant_10000.json --distant-weight 0.50 --hash-size 524288 --context-window 3 --affix-length 5 --outside-weight 0.60 --alpha 1e-6 --max-iter 20 --cache $(NER_CACHE)
RE_ARGS := --distant-train data/distant_10000.json --distant-weight 0.25 --max-relations 2 --relation-no-ratio 0.90 --cache $(RE_CACHE)

.PHONY: dirs train-ner predict-ner eval-ner train-re predict-re eval-re validate

dirs:
	mkdir -p models artifacts predictions results

train-ner: dirs
	$(PYTHON) ner_predict.py data/dev_unlabeled.json predictions/ner_dev.json $(NER_ARGS)

predict-ner: dirs
	$(PYTHON) ner_predict.py data/$(SPLIT)_unlabeled.json predictions/ner_$(SPLIT).json $(NER_ARGS)

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
