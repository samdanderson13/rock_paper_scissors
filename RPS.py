import random

# Reward function
# 1 for a win, 0 for a tie, -1 for a loss
def get_reward(opp_play, user_play):
    if opp_play == user_play:
        return 0
    
    wins = [('R','P'), ('S','R'), ('P','S')]
    if (opp_play, user_play) in wins:
        return 1
    
    else:
        return -1

# Strategy: implements Q-Learning over two state spaces;
#  the opponent's past two moves and the user's past two moves
def player(prev_play,
           opponent_history=[],
           user_history=[],
           Q_opp=dict(),
           Q_user=dict(),
           EPSILON=[],
           EPSILON_BASE=0.9,
           EPSILON_DELTA=0.01,
           MIN_EPSILON=0.1,
           LEARNING_RATE=0.9,
           FUTURE_DISCOUNT=0.3,
           action_table={0: 'R', 1: 'P', 2: 'S'}):

    # Reset for new round
    if not prev_play:
        opponent_history.clear()
        user_history.clear()

        # Initialize Q tables
        Q_opp.clear()
        Q_user.clear()

        for x in action_table.values():
            for y in action_table.values():
                Q_opp[(x,y)] = {'R': 0, 'P': 0, 'S': 0}
                Q_user[(x,y)] = {'R': 0, 'P': 0, 'S': 0}

        EPSILON.clear()
        EPSILON.append(EPSILON_BASE)

    # Pick randomly for the first two turns
    if len(opponent_history) < 2:
        if prev_play != '':
            opponent_history.append(prev_play)
        guess = action_table[random.randrange(3)]
        user_history.append(guess)
        return guess

    # Update Q tables
    prev_user_play = user_history[-1]
    reward = get_reward(prev_play, prev_user_play)

    opp_state = tuple(opponent_history[-2:])
    new_opp_state = (opponent_history[-1], prev_play)

    user_state = tuple(user_history[-3:-1])
    new_user_state = tuple(user_history[-2:])
    
    action = user_history[-1]

    reward_from_opp_table = Q_opp[opp_state][action]
    reward_from_user_table = Q_user[user_state][action]

    future_rewards = []
    for x in Q_opp[new_opp_state].values():
        future_rewards.append(x)
    for y in Q_user[new_user_state].values():
        future_rewards.append(y)
        
    highest_future_reward = max(future_rewards)

    Q_opp[opp_state][action] += LEARNING_RATE * (reward + (FUTURE_DISCOUNT * highest_future_reward) - reward_from_opp_table)
    Q_user[user_state][action] += LEARNING_RATE * (reward + (FUTURE_DISCOUNT * highest_future_reward) - reward_from_user_table)

    opp_state = new_opp_state
    user_state = new_user_state
    opponent_history.append(prev_play)

    # Choose next move
    if random.random() < EPSILON[0]:
        guess = action_table[random.randrange(3)]
    else:
        moves_from_opp_state = Q_opp[opp_state].items()
        moves_from_user_state = Q_user[user_state].items()

        possible_moves = list(moves_from_opp_state) + list(moves_from_user_state)
        guess = max(possible_moves, key=lambda x: x[1])[0]
    
    user_history.append(guess)
    EPSILON[0] = max(EPSILON[0] - EPSILON_DELTA, MIN_EPSILON)
    return guess
