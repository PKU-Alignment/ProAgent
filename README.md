# ProAgent: Building Proactive Cooperative Agents with Large Language Models

## Getting Started

### How to setup
We highly recommend to use Linux or WSL to run instead of Windows, especially you should VPN to link OpenAI API.
It is useful to setup a conda environment with Python 3.7 (python3.8 not support tf1 which is the precondition package used in the baselines):
- Install requirements
    - Do it directly by `conda env create --file environment.yaml`, which will automatically install the conda env, and all Python dependencies in it.
    - Another choice:
        ```
        conda create -n proagent python=3.7
        conda activate proagent

        pip install -r requirements.txt 
        conda install mpi4py==3.1.4 # pip install often fails
        ```

    Notes: Here we use `tf1-cpu`. If you are only familiar with `pytorch`. Don't worry! The baseline models are trained with tf1, so we have to use tf1 to load the model and we ONLY use tf1 when loading the models.

- Install the game environment `overcooked_ai` and another supporting package `stable_baselines` locally.
    ```
    cd ./lib/overcooked_ai
    pip install -e .

    
    cd ./lib/stable_baselines
    pip install -e .
    ```
    Notes 1: Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game Overcooked. https://github.com/HumanCompatibleAI/overcooked_ai.
    Notes 2: `stable_baselines` package is only used for supporting `BC` method, which is trained through the GAIL in `stable_baselines`. 


- Download the baselines. We now support baselines: Greedy, Random, Behavior Cloning(BC), Self-Play(SP),Population-Based Training(PBT), [Fictitious Co-Play (FCP)](https://arxiv.org/abs/2110.08176), [Maximum Entropy Population-Based Training (MEP)](https://arxiv.org/abs/2112.11701), [Cooperative Open-ended Learning (COLE)](https://arxiv.org/abs/2302.04831).
    - Greedy (rule-based) and Random algorithms are built-in methods in the *overcooked_ai* package.
    - For those learning-based algorithms, COLE, FCP, MEP, PBT, SP. Thanks to [COLE-Platform](https://github.com/liyang619/COLE-Platform), we can download those trained models in [here](https://drive.google.com/drive/folders/1s88a_muyG6pVlfcKDKop6R1Fhxr8dcGH) directly, with the structure `layout/model`. 

        - Note: The layout names in code and google drive are not aligned with the layout names in recent papers. Here is the mapping:
            ```
            PYTHON_LAYOUT_NAME_TO_ENV_NAME = {
                "unident_s": "Asymmetric Advantages",
                "simple": "Cramped Room",
                "random1": "Coordination Ring",
                "random0": "Forced Coordination",
                "random3": "Counter Circuit"
            }
            ```
            You need to download those model file and put in the dir ./models.

### How to Run

- Build a `./src/openai_key.txt` file and put your openai key in it. 
- Run `./src/main.py`. For example

    ```
    python main.py --layout cramped_room --p0 Greedy --p1 Greedy --horizon 20
    python main.py --layout coordination_ring --p0 ProAgent --p1 Greedy
    ```
    We support five classical layouts: ['cramped_room', 'asymmetric_advantages', 'forced_coordination', 'coordination_ring', 'counter_circuit'].

- To run many experiments simultaneously, you can use `./src/run.py` where you can select the layouts and algorithms you want to use.



## GPT parser choice

### GPT Model 

- Our main experiments were completed between June and August 2023 based on `gpt-3.5-turbo-0301`, which will be deprecated on June 13th 2024, see: https://platform.openai.com/docs/models/gpt-3-5.

- We also test the `text-davinci-003` and `gpt-4-0314` as the comparsions in our extention version. 
    - However, the model `text-davinci-003` has already been deprecated on Januray 4th 2024 and the official recommended replacement is `gpt-3.5-turbo-instruct`, but we haven't test it yet. 
    - To use GPT-4, please insure your openai key has the right first. As to the first GPT-4 version, model `gpt-4-0314` may be shutdown at earliest 2024-06-13, and the recommended replacement is model `gpt-4-0613`.
    - Learn more here: https://platform.openai.com/docs/deprecations. 

### Prompts level 

- `l1-p`: make plans directly without CoT 
- `l2-ap`: plans with analysis 
- `l3-aip`: plans with analysis and intention


### Memory length
We now support retrival recent K dialogs and use BERT+cos to retrival top K similarity dialogs
```
# without any in-context in the query
python main.py --layout cramped_room --p0 ProAgent --p1 Greedy --retrival_method recent_k --K 0

# with recent 10 dialogs in the query
python main.py --layout cramped_room --p0 ProAgent --p1 Greedy --retrival_method recent_k --K 10
```

> Notes: Our full experiments (5 baselines, 5 layouts) are based on `--gpt_model gpt-3.5-turbo-0301 --prompt_level l2-ap --retrival_method recent_k --K 0or1`. 
We did an ablation study on the impact of choosing different prompts based on `crampt room` layout.
we also use the different LLMs on five layouts only cooperate with Greedy method.


## Q&A 

> [OPENAI ERROR]: Rate limit reached for default-gpt-3.5-turbo in organization org-xxxx tokens per min. Limit: 90000 / min. Current: 88214 / min. Contact us through our help

> [OPENAI ERROR]: Error communicating with OpenAI: HTTPSConnectionPool(host='api.openai.com', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by       
SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1131)')))
>
> `Answer`: Check your VPN connection.

> [OPENAI ERROR]: You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer
auth (i.e. Authorization: Bearer YOUR_KEY), or as the password field (with blank username) if you're accessing the API
from your browser and are prompted for a username and password.
>
> `Answer`: Please delete the blank lines in your `openai_key.txt` file. For example, if you have only one key, just make the file only one line.

> pkg_resources.DistributionNotFound: The 'overcooked_ai' distribution was not found and is required by the application
>
> `Answer`: You may forget to install the `overcooked_ai` package.


## Citation

```bibtex
@inproceedings{zhang2024proagent,
  title={Pro{A}gent: Building Proactive Cooperative Agents with Large Language Models},
  author={Zhang, Ceyao and Yang, Kaijie and Hu, Siyi and Wang, Zihao and Li, Guanghe and Sun, Yihang and Zhang, Cheng and Zhang, Zhaowei and Liu, Anji and Zhu, Song-Chun and Chang, Xiaojun and Zhang, Junge and Yin, Feng and Liang, Yitao  and Yang, Yaodong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17591--17599},
  year={2024}
}
```
