import time
import datetime
import os
import json
from argparse import ArgumentParser
import numpy as np
from rich import print as rprint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*") # ignore "Unable to register cuBLAS factory" due to use tf-CPU


from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

# import pkg_resources
# VERSION = pkg_resources.get_distribution("overcooked_ai").version
import importlib_metadata
VERSION = importlib_metadata.version("overcooked_ai")
print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action


from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent


def main(variant):

    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']


    mode = variant['mode']
    
    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == '0.0.1':
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    
    p0_algo = variant['p0']
    p1_algo = variant['p1']
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")


    start_time = time.time()
    results = []

    for i in range(episode):  

        agents_list = []
        for alg in [p0_algo, p1_algo]:
            if alg == "ProAgent":
                assert variant['gpt_model']!=None, print(f'you should choose a gpt model')
                print(f"\n----Use {variant['gpt_model']}----\n")
                gpt_model = variant['gpt_model']
                retrival_method = variant['retrival_method']
                K = variant['K']
                prompt_level = variant['prompt_level']
                belief_revision = variant['belief_revision']
                agent = make_agent(alg, mdp, layout, model=gpt_model, 
                                   prompt_level=prompt_level, 
                                   belief_revision=belief_revision, 
                                   retrival_method=retrival_method, K=K)
            elif alg == "BC":
                agent = make_agent(alg, mdp, layout, seed_id=i)
            else:
                agent = make_agent(alg, mdp, layout)
            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0

        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                print(env.mdp.state_string(s_t).replace('ø', 'o'))   

                a_t = team.joint_action(s_t) 
                print(f"\n-----------Controller-----------\n")    
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")

                obs, reward, done, env_info = env.step(a_t)

                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')

            ## finish one episode
            if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
                print(f"\n================\n")
                try: # ProAgent id = 0
                    print(f"P1's real behavior: {team.agents[0].teammate_ml_actions_dict}")
                    print(f"The infered P1's intention: {team.agents[0].teammate_intentions_dict}")
                except: # ProAgent id = 1
                    print(f"P0's real behavior: {team.agents[1].teammate_ml_actions_dict}")
                    print(f"The infered P0's intention: {team.agents[1].teammate_intentions_dict}")
                print(f"\n================\n")

            
        elif mode == 'demo':
            pass
         
        print(f"Episode {i+1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)

        
   
    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")



    result_dict = {
        "input": variant,
        "raw_results": results,
        "mean_result": int(np.mean(results)),
    }
    for (k,v) in result_dict.items():
        print(f'{k}: {v}')

    if variant['save']:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        if variant['log_dir'] == None and variant['debug']:
            log_dir = f"experiments/{timestamp}_{layout}_{horizon}_{p0_algo}_{p1_algo}_{episode}numep"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            log_dir = variant['log_dir']

        print(f"This is {log_dir}")
        if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
            json_file = f"{log_dir}/results_{episode}_{horizon}_{gpt_model}_{prompt_level}_{retrival_method}_{K}.json"
        else:
            json_file = f"{log_dir}/results_{episode}_{horizon}.json"
        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=4)


    
if __name__ == '__main__':

    '''
    python main.py --layout cramped_room --p0 Greedy --p1 Greedy --horizon 100
    python main.py --layout cramped_room --p0 ProAgent --p1 BC --horizon 400 -pl l2-ap
    '''
    parser = ArgumentParser(description='OvercookedAI Experiment')

    # these are basis parses
    parser.add_argument('--layout', '-l', type=str, default='cramped_room', choices=['cramped_room', 'asymmetric_advantages', 'coordination_ring', 'forced_coordination', 'counter_circuit'])
    parser.add_argument('--p0',  type=str, default='Greedy', choices=['ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay', 'Human'], help='Algorithm for P0 agent 0')
    parser.add_argument('--p1', type=str, default='Greedy', choices=['ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay', 'Human'], help='Algorithm for P1 agent 1')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon steps in one game')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')

    # these parsers are only required when using ProAgent.
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', choices=['text-davinci-003', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314'], help='Number of episodes')
    parser.add_argument('--prompt_level', '-pl', type=str, default='l2-ap', choices=['l1-p', 'l2-ap', 'l3-aip'], help="'l1-p': make plans directly without CoT; 'l2-ap': plans with analysis; 'l3-aip': plans with analysis and intention.")
    parser.add_argument('--belief_revision', '-br', type=boolean_argument, default=False, help='whether we use belief_revision or not')
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'], help='Use similarity-based(BERT, CLIP) retrieval or retrieve recent K history in dialog.')
    parser.add_argument('--K', type=int, default=1, help="The number of dialogues you want to retrieve.")

    # 
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'demo'], help='exp mode run step-by-step, demo mode run via traj')                                
    parser.add_argument('--save', type=boolean_argument, default=True, help='Whether save the result')
    parser.add_argument('--log_dir', type=str, default=None, help='dir to save result')
    parser.add_argument('--debug', type=boolean_argument, default=True, help='debug mode')


    args = parser.parse_args()
    variant = vars(args)


    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
