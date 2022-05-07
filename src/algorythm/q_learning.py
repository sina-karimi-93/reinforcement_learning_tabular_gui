from inspect import Parameter

from pprint import pprint
import numpy as np

from src.tools.tools import BiggerThanOneDescriptor, BetweenOneAndZero


class Parameters:

    """
    This class is responsible for holding parameters of the program.

    Parameters
    ===================================================================================
    epsilon: Stands for amount of exploration and exploitation. Bigger value of epsilon
    leads to much exploration and vice versa.\n

    alpha: Known as step size.\n
    gamma: known as discount which make the importance of early rewards more valuable.
    it is between 0 and 1 which higher value of this make reinforcement learning process
    more far-sight and vice versa.\n

    episode_count: number episodes which we want to agent try to learn.\n

    model: model of environment contains states and reward of each state.\n

    actions: all actions which agent can use to interact with environment.\n

    x_goal: row position of the goal.\n

    y_goal: column position of the goal.\n

    total_reward: stands for overal rewards which agent gained through learning.\n

    wall_penalty: A penalty which agent receives when collide with the wall.\n

    action_reward: to prevent agent do gratuitous move, we set a small penalty for this.
    """

    x_goal = BiggerThanOneDescriptor()
    y_goal = BiggerThanOneDescriptor()
    rows_count = BiggerThanOneDescriptor()
    columns_count = BiggerThanOneDescriptor()
    epsilon = BetweenOneAndZero()
    alpha = BetweenOneAndZero()
    gamma = BetweenOneAndZero()

    def __init__(self, ) -> None:

        self.model = None
        self.wall_penalty = None
        self.action_reward = None

        self.episode_count = None
        self.actions = ("U", "D", "L", "R")
        self.total_reward = 0

        self.init_epsilon = None

    def make_model(self) -> None:
        """
        Check if any model does not make, make with zeros
        """
        if self.model == None:
            self.model = np.zeros((self.rows_count, self.columns_count))

    def arrow_maker(self, direction) -> str:
        """
        Return correspond arrow for each action.
        """
        directions = {
            "U": "↑",
            "D": "↓",
            "L": "←",
            "R": "→",
        }
        return directions[direction]


class QLearning:
    """
    Q-Learning class is the core of the program which consist
    of train and test methods for Reinforcement Learning process.
    """

    def __init__(self, parameters: Parameters) -> None:

        self.parameters = parameters
        self.Q = self.generate_Q()

    def generate_Q(self) -> np.array:
        """
        This method generate Q based on the number of indexes in model
        and action. For each model's indexes (states) we generate an array of zeros 
        with number of actions.

        example:
            action = ["L","R","U","D"]
            model = [[0,1], [0,1]]
            Q = [ 
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
            ]
        """

        # e.g. 4 x 4
        model_shape = self.parameters.model.shape[0] * \
            self.parameters.model.shape[1]
        Q = np.zeros(
            (model_shape, len(self.parameters.actions)), dtype=int)
        return Q

    def generate_state(self) -> tuple:
        """
        This method generate the state of the agent for each episode.
        state_x       -> agent state in a row (X)
        state_y       -> agent state in a column (Y)
        new_state     -> state number in Q
        """
        x_state = np.random.randint(0, int(self.parameters.model.shape[0] / 2))
        y_state = np.random.randint(0, int(self.parameters.model.shape[0] / 2))
        new_state = (x_state * self.parameters.model.shape[0]) + y_state

        return x_state, y_state, new_state

    def is_reached_the_goal(self, x_state, y_state) -> bool:
        """
        Check whether the agent reach the goal or not
        """
        is_reached = self.parameters.model[x_state,
                                           y_state] != self.parameters.model[self.parameters.x_goal-1, self.parameters.y_goal-1]
        return is_reached

    def reduce_exploration(self, episode) -> float:
        """
        To more exploit rather than explore we have to decrease the
        epsilon during the training.
        """

        self.parameters.epsilon *= (self.parameters.episode_count -
                                    episode) / self.parameters.episode_count

    def check_action_result(self, action, x_state, y_state) -> tuple:
        """

        """

        reward = self.parameters.action_reward
        if action == "L":
            if y_state == 0:  # left wall
                reward += self.parameters.wall_penalty
            else:
                y_state -= 1

        elif action == "R":
            if y_state == (self.parameters.model.shape[1]-1):  # right wall
                reward += self.parameters.wall_penalty
            else:
                y_state += 1

        elif action == "U":
            if x_state == 0:  # north wall
                reward += self.parameters.wall_penalty
            else:
                x_state -= 1

        elif action == "D":  # south wall
            if x_state == self.parameters.model.shape[0]-1:
                reward += self.parameters.wall_penalty
            else:
                x_state += 1

        reward += self.parameters.model[x_state, y_state]
        return x_state, y_state, reward

    def update_policy(self, *, current_state, new_state, reward, action_index):
        """
        After the agent movement, the states and rewards are change. So we should update
        the values of the Q and renew the policy.
        In the following formula new policy will acheive. (Bellman Equation)
        """
        self.Q[current_state, action_index] += (self.parameters.alpha *
                                                (reward + (self.parameters.gamma *
                                                 self.Q[new_state].max()) - self.Q[current_state, action_index]))

    def train(self) -> None:
        """
        Process of agent learning starts here.
        """
        for episode in range(self.parameters.episode_count):

            # For each episode we randomly introduce the state
            # new_state -> row number in Q
            x_state, y_state, new_state = self.generate_state()

            # At first epsilon is equal to 0.9 to agent explore more than exploiting
            # But we have to reduce the epsilon throughout the game to more exploiting
            self.reduce_exploration(episode=episode)

            # Number of tries to reaching goal in each episode
            try_per_episode = 0

            # Check if agent reached the goal or 1000 times tries to reach the goal
            while self.is_reached_the_goal(x_state, y_state) and try_per_episode < 1000:
                try_per_episode += 1

                current_state = new_state

                # choose action greedyly (based on most valuabe action)
                random_number = np.random.random()  # 0 < random_number < 1
                if random_number > self.parameters.epsilon:
                    action_index = self.Q[current_state].argmax()

                # choose action act e-greedy (random)
                else:
                    action_index = np.random.randint(
                        0, len(self.parameters.actions))

                # select action
                action = self.parameters.actions[action_index]

                # After agent's action we have to update the state and calculate the reward
                # gained based on that action
                x_state, y_state, reward = self.check_action_result(
                    action=action, x_state=x_state, y_state=y_state)

                # update state number after action
                new_state = (
                    x_state * self.parameters.model.shape[0]) + y_state

                # add gained reward to total reward
                self.parameters.total_reward += reward

                # update policy
                self.update_policy(
                    current_state=current_state,
                    new_state=new_state,
                    reward=reward,
                    action_index=action_index)
                # end of try per episode
            # end of episode

    def test(self) -> tuple:
        """
        After training, The Q gives us a lot of
        information about the best action in each state.
        So we extract them into to arrays policy and value_state.
        """
        state_counts = self.parameters.model.shape[0] * \
            self.parameters.model.shape[1]

        self.policy = np.array([
            'o' for _ in range(state_counts)])

        self.value_state = np.zeros(
            (self.parameters.model.shape[0] * self.parameters.model.shape[1]))

        for s in range(self.Q.shape[0]):
            qs = self.Q[s]
            action_index = qs.argmax()
            self.policy[s] = self.parameters.actions[action_index]
            self.value_state[s] = qs.max()

        return self.policy, self.value_state

    def get_results(self) -> tuple:
        """
        Return the policy, value_state and Q
        """
        self.test()
        self.parameters.policy = list(self.policy)
        self.parameters.value_state = list(self.value_state)
        return {"policy": self.policy,
                "value_state": self.value_state,
                "Q": self.Q,
                "total_reward": self.parameters.total_reward}
