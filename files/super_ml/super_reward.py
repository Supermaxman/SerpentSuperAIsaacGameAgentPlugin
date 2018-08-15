

def create_rewards(game_state, game_frame):
    move_reward = 0.0
    attack_reward = 0.0
    if game_state["isaac_alive"]:
        if game_state["damage_taken"]:
            damage_taken = game_state["isaac_hps"][1] - game_state["isaac_hps"][0]
            move_reward += -1.0 * damage_taken
        else:
            if game_state["damage_dealt"]:
                move_reward += 0.05
                attack_reward += 0.05
            else:
                move_reward += 0.01
        if game_state["boss_dead"]:
            move_reward += 1.0
            attack_reward += 1.0
    else:
        move_reward += -2.0
    return move_reward, attack_reward