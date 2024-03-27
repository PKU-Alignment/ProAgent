import random
import time

import numpy as np
import openai


from overcooked_ai_py.mdp.actions import Action, Direction
import numpy as np 


def convert_messages_to_prompt(messages):
    """
    Converts a list of messages(for chat) to a prompt (for completion) for OpenAI's API.

    :param messages:
    :return: prompt
    """
    prompt = ""
    for message in messages:
        prompt += f"{message['content']}\n"

    return prompt


def gpt_state_list(mdp, state):
    """List representation of the current state, modified for GPT."""
    players_dict = {player.position: player for player in state.players}
    grid_list = []

    for y, terrain_row in enumerate(mdp.terrain_mtx):
        grid_row = []
        for x, element in enumerate(terrain_row):
            if (x, y) in players_dict.keys():
                player = players_dict[(x, y)]
                orientation = player.orientation
                player_object = player.held_object
                assert orientation in Direction.ALL_DIRECTIONS
                player_idx_lst = [
                    i
                    for i, p in enumerate(state.players)
                    if p.position == player.position
                ]
                assert len(player_idx_lst) == 1

                if player_object:
                    if player_object.name[0] == "s":
                        # this is a soup
                        grid_row.append("{}-{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation], str(player_object)))
                    else:
                        grid_row.append("{}-{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation], player_object.name[:1]))
                else:
                    grid_row.append("{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation]))
            else:
                if element == "X" and state.has_object((x, y)):
                    state_obj = state.get_object((x, y))
                    if state_obj.name[0] == "s":
                        grid_row.append(str(state_obj))
                    else:
                        grid_row.append(state_obj.name[:1])
                elif element == "P" and state.has_object((x, y)):
                    soup = state.get_object((x, y))
                    grid_row.append(element+str(soup))
                else:
                    grid_row.append(element)

        grid_list.append(grid_row)

    if state.bonus_orders:
        bonus_orders = ["Bonus orders: {}".format(state.bonus_orders)]
        grid_list.append(bonus_orders)

    return grid_list


def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.error.RateLimitError,),
):

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                print(e)
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper