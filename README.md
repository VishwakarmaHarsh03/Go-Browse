# Go-Browse
Automatic, unsupervised collection of web agent training data via structured exploration of websites.

## Setup

Note, we ran our experiments with Python 3.12, though earlier python versions may also work.

1. Follow the instructions here to install browsergym with webarena and playwright with chromium: https://github.com/ServiceNow/BrowserGym
2. Install `webexp` and dependencies:
```sh
pip install -r requirements.txt
pip install -e .
```
3. Setup a WebArena Server using the instructions here: [webarena readme](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md). You can also optionally setup the a reset server to remotely reset the webarena environments by: 
    - Copy/clone over the webarena-reset folder to your webarena hosting instance
    - `pip install fastapi[standard]` on this instance.
    - `cd webarena-reset`
    - `export BASE_URL=<PUBLIC URL for your instance>`
    - `fastapi run reset_server.py`
    - You can now reset a specific domain at once (e.g. map with `<RESET_SERVER_URL>/reset/map`) or all domains at once with (e.g., `<RESET_SERVER_URL>/reset/all`).

## Collect Dataset
Example config file used for Go-Browse-WA data generation is: `configs/go_browse_config.yaml`

For each domain (website) that you want to run data generation for, duplicate/modify the config file by filling in placeholders and then run:
```sh
python -m webexp.explore.algorithms.web_explore -c configs/web_explore_config.yaml
```

### Process Collected Go-Browse Dataset for Training
First, set the input and output paths as appropriate in `projects/go-browse/data/generate_dataset.py` and `projects/go-browse/data/process_dataset.py`

Then:
```sh
python projects/go-browse/data/generate_dataset.py
python projects/go-browse/data/process_dataset.py
```

### Process NNetNav Dataset for Training
First, set the output path as appropriate in `projects/go-browse/data/process_nnetnav_data.py`

Then:
```
python projects/go-browse/data/process_nnetnav_data.py
```

## Finetune a Model
First, replace the placeholder paths/env vars as appropriate in `webexp/train/sft_policy.py`

Then:
```
python webexp/train/sft_policy.py
```

## Benchmark a Model on WebArena
If benchmarking a finetuned model, first serve the model using an inference server like [vllm](https://docs.vllm.ai/en/latest/) or [sglang](https://docs.sglang.ai/). We used `vllm` in our experiments.

Duplicate/edit the following config file by filling in the placeholders: `configs/benchmark_webarena.yaml`.

Then:
```
python -m webexp.benchmark.run_webarena -c configs/benchmark_webarena.yaml
```

## Run an Episode on a Website
If performing inference with a finetuned model, first serve the model using an inference server like [vllm](https://docs.vllm.ai/en/latest/) or [sglang](https://docs.sglang.ai/).

Duplicate/edit the following config file by filling in the placeholders: `configs/benchmark_webarena.yaml`.

Then:
```
python -m webexp.agents.run_episode -c configs/benchmark_webarena.yaml
```

## Go-Browse-WA Dataset and Trained Models Release

Here is a link to a processed version of the dataset used for our finetuning results (output of `projects/go-browse/data/process_dataset.py`): [go-browse-wa-processed.jsonl](https://drive.google.com/file/d/1yqrFBybA6YerxlOQvHdoXOM4gC_Au7QX/view?usp=sharing).
This dataset includes both successful and unsuccessful trajectories. All experiments from the paper can be reproduced with this version of the dataset by filtering to just the successful trajectories.

Trained model checkpoints and full raw dataset (including alternate observations representations like screenshots and pruned html) will be released in the future.
