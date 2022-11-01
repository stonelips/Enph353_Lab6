import random
import pickle
import math


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        random.seed()

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print("Loaded file: {}".format(filename+".pickle"))
        return f

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self.q, f)

        # TODO: CSV files.

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)
        
    def initiateQ(self, state, action):
        try:
            self.q[(state, action)]
        except KeyError:
            # Initiate with zero reward if [state, action] DNE
            self.q[(state, action)] = 0.0 
    

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action
        
        random.seed()
        diceroll = random.random()

        if (diceroll < self.epsilon):
            random.seed()
            action = random.choice(self.actions)

            self.initiateQ(state, action)
            q = self.q[(state, action)]
        else:
            action = 0
            q = -1E6
            for a in self.actions:
                self.initiateQ(state, a)

                if self.q[(state, a)] > q:
                    action = a
                    q = self.q[(state, a)]
                # If two actions have same maxQ, 50% chance of choosing a new action.
                elif self.q[(state, a)] == q:
                    if (math.floor(random.random() * 10) % 2 == 0):
                        action = a
                        q = self.q[(state, a)]
        
        
        return (action,q) if return_q==True else action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        self.initiateQ(state1, action1)
        q1 = self.q[(state1, action1)]

        # action, actionReward = self.chooseAction(state2, True)

        actionReward = -1E6 
        for a in self.actions:
            self.initiateQ(state2, a)
            if self.q[(state2, a)] > actionReward:
                actionReward = self.q[(state2, a)]

        self.q[(state1, action1)] = q1 + self.alpha*(reward + self.gamma*actionReward - q1)
