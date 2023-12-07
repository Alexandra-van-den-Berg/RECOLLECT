"""
RECOLLECT architecture. @author: A.R. van den Berg
Based on AuGMEnT implementation @author: J.O. Rombouts
"""

import numpy as np
from numpy.random import rand
import pickle as pickle
import logging


class Sigmoid(object):
    """
    Simple container class for sigmoid transformation.
    """

    def __init__(self, theta=2.5, sig_slope=1.):
        self.theta = theta
        self.sig_slope = sig_slope

    def transform(self, np_arr):
        return 1. / (1. + np.exp(self.theta - (self.sig_slope * np_arr)))

    @staticmethod
    def derivative(np_arr):
        return np_arr * (1. - np_arr)


class RECOLLECTNetwork(object):
    """
    Implementation of RECOLLECT network
    """

    def __init__(self, **kwargs):
        """
        Constructor. See comments below for explanation of kwargs.
        """

        # Network meta-parameters
        self.beta = kwargs.get('beta', 0.25)  # Learning rate
        self.beta_gate = self.beta * kwargs.get('learn_gate', 0.25) 
        self.gamma = kwargs.get('gamma', 0.90)  # Future discounting

        # Parameters controller:
        self.epsilon = kwargs.get('epsilon', 0.025)  # Exploration rate
        self.explore = self.select_uniform_random # Exploration fn. to use
        self.explore_nm = 'e-greedy'
        self.prec_q_acts = kwargs.get('prec_q_acts', 5)  # Cutoff of q-values

        # Architecture Parameters
        self.nx_inst = kwargs.get('n_inputs', 4)  # Number input neurons (Instantaneous)
        self.ny_memory = kwargs.get('ny_memory', 4)  # Number of memory hidden units
        self.nz = kwargs.get('nz', 3)  # Number of output units

        self.weight_range = kwargs.get('weight_range', 0.25)  # Centered range of weights
        self.L = kwargs.get('L', 0.2)

        self.theta = kwargs.get('theta', 2.5)  # Sigmoid shift
        self.sig_slope = kwargs.get('s_slope', 1.) # Sigmoid slope
        self.cell_transform =  Sigmoid(theta=self.theta, sig_slope=self.sig_slope)
        self.gate_transform = Sigmoid(theta=self.theta, sig_slope=self.sig_slope)
        self.mem_transform = Sigmoid(theta=self.theta)

        # Synaptic weights:
        # Bias units (set to 0 to switch off)
        self.bias_input = kwargs.get('bias_input', 1)
        self.bias_mem_hidden = kwargs.get('bias_mem_hidden', 1)

        # Input to hidden weights (gating units)
        self.weights_xy_gate = self.generate_weights(self.nx_inst + self.bias_input, self.ny_memory) 

        # Input to hidden weights (candidate input units)
        self.weights_xy_cell = self.generate_weights(self.nx_inst + self.bias_input, self.ny_memory) 

        # Memory unit to output unit
        self.weights_yz_mem = self.generate_weights(self.ny_memory + self.bias_mem_hidden, self.nz)


        # Dynamic Network Parameters
        self._initialize_dynamic_variables()

        logging.getLogger(__name__).debug("Initialized Network.")

    def _initialize_dynamic_variables(self):
        """
        Sets up all dynamic variables
        """
        # Layer activations:
        self.x_reg = np.zeros(self.nx_inst) 
        self.gate = np.zeros(self.ny_memory)
        self.cell = np.zeros(self.ny_memory)
        self.y_mem = np.zeros(self.ny_memory)
        self.prev_mem = np.zeros(self.ny_memory)
    
        self.z = np.zeros(self.nz)

        # Input to hidden traces:
        self.xy_gate_traces = np.zeros_like(self.weights_xy_gate)
        self.xy_cell_traces = np.zeros_like(self.weights_xy_cell)
        self.yz_mem_traces = np.zeros_like(self.weights_yz_mem)

        # Tags (~ eligibility traces)
        self.xy_cell_tags = np.zeros_like(self.weights_xy_cell)
        self.xy_gate_tags = np.zeros_like(self.weights_xy_gate)
        self.yz_mem_tags = np.zeros_like(self.weights_yz_mem)


        self.prev_obs = np.zeros(self.nx_inst)
        self.z_prime = np.zeros(self.nz)
        self.prev_action = -1
        self.prev_qa = None
        self.delta = 0


    def do_step(self, observation, reward):
        """
        This function handles the interaction with Tasks, by taking
        (observation, reward) and selecting an action.

        :param: observation: current output from Task
        :param: reward: scalar reward obtained from Task

        :return: selected action (one-hot encoded)
        """
        logging.getLogger(__name__).debug("reward: {}, observation: {}".format(reward, observation))

        self.compute_input(observation)
        self.compute_hiddens()
        self.compute_output()

        self.select_action()

        self.exp_val = self.z[self.prev_action]
        logging.getLogger(__name__).debug("exp val: {}".format(self.exp_val))

        # Compute TD-error
        if self.prev_qa is None:
            # No previous expectation. Set to current expectation.
            self.prev_qa = self.exp_val

        # # Calculate TD-error:
        self.delta = self.compute_delta(reward)
        logging.getLogger(__name__).debug("delta: {}".format(self.delta))

        self.update_weights()
        self.update_traces()
        self.update_tags()

        # Set previous observation to current
        self.prev_obs = self.x_reg
        self.prev_qa = self.exp_val
        self.prev_mem = np.array(self.y_mem)

        # Return action (1-hot encoded vector)
        return self.z_prime


    def compute_input(self, observation):
        """
        Compute input to the network (takes care of transient units)
        :param: observation: current output from Task
        """

        self.x_reg = observation


    def compute_hiddens(self):
        """
        Compute activations of hidden units.
        """
        self.compute_gate()
        self.compute_cell()
        self.compute_memory()


    def compute_gate(self):
        # Linear sum
        gate_acts = np.dot(np.hstack((np.ones(self.bias_input), self.x_reg)), self.weights_xy_gate)

        # Transform
        self.gate = self.gate_transform.transform(gate_acts)


    def compute_cell(self):
        # Linear sum
        cell_acts = np.dot(np.hstack((np.ones(self.bias_input), self.x_reg)), self.weights_xy_cell)

        # Transform
        self.cell = self.cell_transform.transform(cell_acts)


    def compute_memory(self):
        self.y_mem = ((self.gate * self.prev_mem) + ((1-self.gate) * self.cell))
    

    def compute_output(self):
        """Q unit activations"""
        self.z = np.dot(np.hstack((np.ones(self.bias_mem_hidden), self.y_mem)), self.weights_yz_mem)
        self.z = np.round(self.z, self.prec_q_acts)


    def select_action(self):
        """
        Select actions based on Q-unit activations and selected controller.
        """
        max_q = np.max(self.z)

        # Determine whether to make exploratory move:
        if np.random.sample() <= self.epsilon:
            action = self.explore(self.z)
        else:
            # Take greedy action (break ties randomly):
            idces = np.where(self.z == max_q)[0]
            if np.size(idces) > 1:
                action = idces[np.random.randint(0, np.size(idces))]
            else:
                action = idces[0]

        # Set output-action:
        self.z_prime = np.zeros(self.nz)
        self.z_prime[action] = 1
        self.prev_action = action
        logging.getLogger(__name__).debug("action: {}".format(self.z_prime))

    @staticmethod
    def compute_softmax(values):
        """
        Compute softmax transformation
        """
        # Trick for numerical stability
        values = values - np.max(values)

        # Pull result through softmax operator:
        exps = np.exp(values)
        values = exps / np.sum(exps)

        return values

    @staticmethod
    def select_boltzmann(values):
        """
        Select random action from Boltzmann distribution
        by Roulette wheel selection
        """
        boltz = RECOLLECTNetwork.compute_softmax(values)

        # Create wheel:
        probs = [sum(boltz[:i + 1]) for i in range(len(boltz))]

        # Select from wheel
        rnd = np.random.sample()
        for (i, prob) in enumerate(probs):
            if rnd <= prob:
                return i

    def select_uniform_random(self, values):
        """
        select uniform random action
        """
        return np.random.randint(0, self.nz)

    def compute_delta(self, reward):
        """
        Compute SARSA TD error.
        """
        return (reward + (self.gamma * self.exp_val) - self.prev_qa)

    
    def update_weights(self):
        """
        Do weight updates
        """

        self.weights_yz_mem += (self.beta * self.delta * self.yz_mem_tags)

        self.weights_xy_cell += (self.beta * self.delta * self.xy_cell_tags)

        self.weights_xy_gate += (self.beta_gate * self.delta * self.xy_gate_tags)


    def update_traces(self):
        """
        Update the traces. Note that these are only located between the input and hidden layer
        """

        cell_inp = np.tile(np.hstack((np.ones(self.bias_input), self.x_reg)), self.ny_memory).reshape(self.ny_memory,self.bias_input+self.nx_inst).T

        gate_inp = np.tile(np.hstack((np.ones(self.bias_input), self.x_reg)), self.ny_memory).reshape(self.ny_memory,self.bias_input+self.nx_inst).T

        self.xy_cell_traces = (((self.gate * self.xy_cell_traces) + (1-self.gate) * cell_inp * self.cell_transform.derivative(self.cell)))

        self.xy_gate_traces = ((self.gate * self.xy_gate_traces) + ((self.prev_mem - self.cell) * gate_inp * self.gate_transform.derivative(self.gate)))

    
    def update_tags(self):
        """
        Compute the updates for the tags (~eligibity traces). Traces are the
        records of feedforward activation that went over the synapses to
        postsynaptic hidden neurons.
        """

        # 1. Decay old tags:

        # Input to hidden:
        self.xy_cell_tags = self.xy_cell_tags * self.L * self.gamma

        self.xy_gate_tags = self.xy_gate_tags * self.L * self.gamma

        # Hidden to output:
        self.yz_mem_tags = self.yz_mem_tags * self.L * self.gamma

        # 2. Update tags:

        # # Output to hidden:
        self.yz_mem_tags[:, self.prev_action] += np.hstack((np.ones(self.bias_mem_hidden), self.y_mem))

        # # Input to hidden:
        # # Here feedback and traces interact to form tag update:

        # # Feedback from output layer to memory hidden units:
        fb_mem = self.weights_yz_mem[self.bias_mem_hidden:, self.prev_action]

        # # Cell and gate updates 
        self.xy_cell_tags += (self.xy_cell_traces * fb_mem)

        self.xy_gate_tags += (self.xy_gate_traces * fb_mem)


    def set_learning(self, state='off'):
        """
        Turn learning off / on
        """
        if state == 'off':
            self.__beta = self.beta
            self.beta = 0.
        else:
            self.beta = self.__beta


    def set_exploration(self, state='off'):
        """
        Turn exploration off / on
        """
        if state == 'off':
            self.__epsilon = self.epsilon
            self.epsilon = 0.
        else:
            self.epsilon = self.__epsilon


    def generate_weights(self, n_in, n_out):
        """
        Generate random (n_in, n_out) weight matrix in range self.weight_range.
        """
        weights = (2. * self.weight_range) * rand(n_in, n_out) - self.weight_range

        return weights


    def __getstate__(self):
        """
        Determines class attributes that get pickled to file
        """
        odict = self.__dict__.copy()  # copy the dict since we change it

        # Remove methods, as they can not be saved to a file
        del odict['explore']  # Note: this is restored by explore_nm - brittle
        del odict['gate_transform']
        del odict['cell_transform']

        return odict

    def __setstate__(self, dict):
        """
        Reconstruct RECOLLECT object from pickled dict
        """
        if dict['explore_nm'] == 'max-boltzmann':
            dict['explore'] = self.select_boltzmann
        elif dict['explore_nm'] == 'e-greedy':
            dict['explore'] = self.select_uniform_random

        dict['gate_transform'] = Sigmoid(theta=dict['theta'], sig_slope=dict['sig_slope'])
        dict['cell_transform'] = Sigmoid(theta=dict['theta'], sig_slope=dict['sig_slope'])

        # Update dictionary:
        self.__dict__.update(dict)


    def save_network(self, path_to_file):
        """
        Pickle the network object to a file. Note that this is brittle, it just pickles the objects' self.__dict__
        """
        try:
            fl = open(path_to_file, 'wb')
            pickle.dump(self, fl)
            fl.close()
        except IOError as v:
            print("Something went wrong trying to access: {}".format(path_to_file))
            print(v)

