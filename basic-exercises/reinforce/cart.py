import gymnasium as gym
import numpy as np
import random

# ---------------------------------------------------------
# 1. Create the environment
# ---------------------------------------------------------

env = gym.make("CartPole-v1")

# ---------------------------------------------------------
# 2. CartPole gives us 4 continuous numbers:
#
#    cart position
#    cart velocity
#    pole angle
#    pole angular velocity
#
# We turn those continuous numbers into buckets so that
# we can use a simple Q-table.
# ---------------------------------------------------------

NUM_BUCKETS = (10, 10, 10, 10)

# Approximate useful ranges for each observation.
LOW = np.array([
    -2.4,    # cart position
    -3.0,    # cart velocity
    -0.42,   # pole angle
    -3.5     # pole angular velocity
])

HIGH = np.array([
     2.4,
     3.0,
     0.42,
     3.5
])


# ---------------------------------------------------------
# 3. Q-table
#
# Q[state][action] tells us:
#
# "How good is this action when I'm in this state?"
#
# There are:
#
# 10 x 10 x 10 x 10 = 10,000 states
#
# and 2 possible actions.
# ---------------------------------------------------------

q_table = np.zeros(NUM_BUCKETS + (env.action_space.n,))


# ---------------------------------------------------------
# 4. Convert the real-valued observation into a bucket
# ---------------------------------------------------------

def discretize(observation):
    observation = np.clip(observation, LOW, HIGH)
    ratios = (observation - LOW) / (HIGH - LOW)
    buckets = (ratios * np.array(NUM_BUCKETS)).astype(int)
    buckets = np.minimum(buckets, np.array(NUM_BUCKETS) - 1)
    return tuple(buckets)

# ---------------------------------------------------------
# 5. Choose an action
#
# Most of the time:
#     choose the action our Q-table thinks is best.
#
# Sometimes:
#     try something random.
#
# This is exploration vs exploitation.
# ---------------------------------------------------------
def choose_action(state, epsilon):
    if random.random() < epsilon:
        # Explore
        return env.action_space.sample()

    else:
        # Exploit
        return np.argmax(q_table[state])

# ---------------------------------------------------------
# 6. Training parameters
# ---------------------------------------------------------

episodes = 5000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01


# ---------------------------------------------------------
# 7. TRAIN
# ---------------------------------------------------------

for episode in range(episodes):
    observation, info = env.reset()
    state = discretize(observation)
    total_reward = 0
    done = False

    while not done:

        # ---------------------------------------------
        # Agent chooses an action
        # ---------------------------------------------

        action = choose_action(state, epsilon)

        # ---------------------------------------------
        # Environment executes the action
        # ---------------------------------------------

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---------------------------------------------
        # Convert new observation into our state
        # ---------------------------------------------

        new_state = discretize(observation)


        # ---------------------------------------------
        # Q-learning update
        #
        # Q(s,a) =
        #
        # Q(s,a) + learning_rate *
        #     (reward +
        #      discount_factor * max(Q(new_state))
        #      - Q(s,a))
        # ---------------------------------------------

        best_future_value = np.max(q_table[new_state])
        current_value = q_table[state + (action,)]

        if done:
            target = reward
        else:
            target = reward + discount_factor * best_future_value

        q_table[state + (action,)] = (
            current_value
            + learning_rate * (target - current_value)
        )


        # Move to the new state

        state = new_state

        total_reward += reward


    # -------------------------------------------------
    # Reduce exploration over time
    # -------------------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # -------------------------------------------------
    # Print progress
    # -------------------------------------------------

    if episode % 100 == 0:
        print(
            f"Episode: {episode:4d} "
            f"Reward: {total_reward:4.0f} "
            f"Epsilon: {epsilon:.3f}"
        )

# ---------------------------------------------------------
# 8. Training finished
# ---------------------------------------------------------

print()
print("Training complete!")
print()


# ---------------------------------------------------------
# 9. Watch the trained agent
# ---------------------------------------------------------
env.close()

env = gym.make("CartPole-v1", render_mode="human")

for episode in range(5):
    observation, info = env.reset()
    state = discretize(observation)
    done = False
    total_reward = 0

    while not done:

        # No exploration now.
        # Always choose what the Q-table thinks is best.

        action = np.argmax(q_table[state])
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = discretize(observation)
        total_reward += reward

    print(
        f"Test episode {episode + 1}: "
        f"reward = {total_reward}"
    )

env.close()
