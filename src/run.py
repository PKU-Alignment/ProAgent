import os
import time

import subprocess
import multiprocessing
import concurrent.futures

layouts = [
    'cramped_room', #'coordination_ring', 'forced_coordination', 'counter_circuit', 'asymmetric_advantages'
]  

player_algorithms = {
    'player0': ['GPT', ], #'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'GPT'],  
    'player1': ['Greedy', ]  #'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'GPT']   
}


horizon = 10
episode = 1
model="text-davinci-003" 


def run_experiment(layout, model, player0_algo, player1_algo, horizon=400, episode=5):
    result_dir = f"results/{model}/{layout}/{player0_algo}/{player1_algo}"
    os.makedirs(result_dir, exist_ok=True)
    json_file = f"{result_dir}/results_{episode}_{horizon}.json"
    if not os.path.exists(json_file):
        print(f"Start: {layout} | {player0_algo} vs {player1_algo}")
        command = f"python main.py --gptmodel {model} --layout {layout} --p0 {player0_algo} --p1 {player1_algo} --episode {episode} --horizon {horizon} --save True --log_dir {result_dir}"
        with open(f"{result_dir}/run.txt", "w") as f:
            subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return layout, player0_algo, player1_algo 


def main():
    num_processes = multiprocessing.cpu_count()
    print(f"\nNumber of available CPU cores: {num_processes}\n")
    pool = multiprocessing.Pool(processes=num_processes-10)
    for layout in layouts:
        for player0_algo in player_algorithms['player0']:
            for player1_algo in player_algorithms['player1']:
                while True:
                    try:
                        result = pool.apply_async(run_experiment, args=(layout, model, player0_algo, player1_algo, horizon, episode))
                        layout, player0_algo, player1_algo = result.get()  
                        print(f"Completed: {layout} | {player0_algo} vs {player1_algo}\n")
                        break  
                    
                    except TimeoutError:
                        print(f"Experiment timeout: {layout} | {player0_algo} vs {player1_algo}")
    pool.close()
    pool.join()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
