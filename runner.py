"""
Example code to train RECOLLECT on the pro-/anti-saccade task @author: A.R. van den Berg
Code based on AuGMEnT code @author: J.O. Rombouts
"""

from recollect import RECOLLECTNetwork
from task import GGSA 
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd

# Disable plotting if matplotlib is not installed
global PLOTTING
PLOTTING = True
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    PLOTTING = False
    print("Matplotlib is not installed - disabling plotting")


def run_experiment(seed):
    # Define relevant parameters
    learn_r = 0.1 # learning rate
    learn_g = 0.6 # multiplication factor learning rate for gating units
    decay = 0.4 # tag decay rate
    explore_r = 0.025 # exploration rate
    n_mem = 7 # number of hidden units
    theta = 0. # alters sigmoid
    sig_slope = 2. # alters sigmoid
    inter_time = 0 # number of intertrial steps
    mem_delay = 2 # number of memory delay steps
    reset_input = False # use reset signal from task as additional input 
    cur_on = False # whether to use a curriculum to obtain the final memory delay 
    
    # Set random seed:
    np.random.seed(seed)

    # Maximal number of trials to run
    trials = 500000

    # Target proportion correct over all trial types:
    stopping_crit = 0.85

    # Measurement window for stopping criterion
    stopping_crit_window = 100

    # Chooses appropriate memory delay for curriculum step
    if cur_on:
        curr_change = {np.ceil(mem_delay / 5):0}
        current_delay = np.ceil(mem_delay/5)
    else:
        current_delay = mem_delay

    task = GGSA(inter_dur=inter_time, memory_dur=current_delay)

    # Buffers for storing performance for all trial_types
    n_trial_types = len(task.trial_types)
    result_buffer = np.zeros((n_trial_types, stopping_crit_window))
    result_buffer_idces = np.zeros(n_trial_types)
    result_buffer_avgs = np.zeros((n_trial_types, trials))

    # Buffer for curriculum-dependent performance information 
    if cur_on:
        new_curriculum_buffer = np.zeros((n_trial_types, stopping_crit_window))
        new_curriculum_buffer_idces = np.zeros(n_trial_types)
        new_curriculum_buffer_avgs = np.zeros((n_trial_types, trials))

    # Trial results (0: failure, 1: success)
    trial_res = np.zeros(trials)

    # Get size of task output (to determine input layer size)
    new_input = task.display.to_output() 

    # Build network (see RECOLLECTNetwork source for all options)
    if reset_input:
        network = RECOLLECTNetwork(beta=learn_r, L=decay, n_inputs=len(new_input)+1, nz=3, ny_memory=n_mem, 
                                weightrange=.25, gamma=0.9, epsilon=explore_r, learn_gate=learn_g, theta=theta, s_slope = sig_slope)
    else:
        network = RECOLLECTNetwork(beta=learn_r, L=decay, n_inputs=len(new_input), nz=3, ny_memory=n_mem, 
                                weightrange=.25, gamma=0.9, epsilon=explore_r, learn_gate=learn_g, theta=theta, s_slope = sig_slope)

    reward = 0.
    reset = False
    converged = False
    c_epoch = -1
    unstable = False

    if PLOTTING:
        # For live update of plots
        plt.ion()
        g = Graphs(max_trials=trials, n_subtasks=n_trial_types)
        plot_interval = 500
    else:
        logging.getLogger(__name__).warning("Not generating plots - matplotlib is ")

    for trial in range(0, trials):

        # Plotting of network performance
        if PLOTTING and trial % plot_interval == 0:
            logging.getLogger(__name__).info("trial = {:d}".format(trial))

            # Plot performance
            g.update_network_performance(trial_res)

            # Plot performance on all trial types separately
            for i in range(n_trial_types):
                y = result_buffer_avgs[i, :]
                g.update_subtask_performance(i, y)
            g.draw()

        if converged:
            break

        trial_running = True
        while trial_running and not converged and not unstable:
            # Get action from network based on latest output from Task

            # If reset input should be provided, reset is added onto the input
            if reset_input:
                action = network.do_step(np.hstack((new_input, int(reset))), reward)
            else:
                action = network.do_step(new_input, reward)

            # Networks can become unstable with high learning rates and
            # long trace decay times. If this happens, try tuning the parameters
            # Figure 7 in the paper indicates good parameter combinations
            if not check_stable(network.z):
                unstable = True
                print('Network unstable, abort')
                break

            if reset:  # End of trial detected:
                logging.getLogger(__name__).info("Trial {:d} end".format(trial))

                trial_running = False
                tmp_tp = task.last_trial_type

                if reward == task.fin_reward:  # Mark trial as successful
                    logging.getLogger(__name__).info("Obtained reward!")

                    trial_res[trial] = 1

                    # Add result to results for specific input-pattern:
                    result_buffer[tmp_tp, int(result_buffer_idces[tmp_tp])] = 1
                    
                    if cur_on:
                        new_curriculum_buffer[tmp_tp, int(new_curriculum_buffer_idces[tmp_tp])] = 1

                    # Compute convergence for all buffers:
                    # Note that all last stopping_crit_window trials of all types have to be
                    # at convergence criterion.
                    
                    if cur_on:
                        # If the curriculum is used, check performance for the current memory delay. 
                        # If this above criterion performance, then make task more difficult.
                        if np.all(np.average(new_curriculum_buffer, axis=1) > stopping_crit):
                            print('\nAbove stopping criterion for %.0f memory delay steps' % current_delay)
                            if current_delay != mem_delay:
                                curr_change[current_delay] = trial
                                current_delay = np.ceil(current_delay / 0.5)
                                if current_delay > mem_delay:
                                    current_delay = mem_delay
                                curr_change[current_delay] = 0

                                # Reinitialise buffers
                                new_curriculum_buffer = np.zeros((n_trial_types, stopping_crit_window))
                                new_curriculum_buffer_idces = np.zeros(n_trial_types)
                                new_curriculum_buffer_avgs = np.zeros((n_trial_types, trials))

                                task = GGSA(inter_dur=inter_time, memory_dur=current_delay)

                            else: 
                                # Achieved criterion performance on all trial types
                                # Now, check that all patterns can be classified correctly
                                # without exploration (and fixed network-weights)
                                print('\nFinal memory delay reached, checking performance now')
                                if check_performance(network, inter_time, mem_delay, reset_input):
                                    c_epoch = trial
                                    
                                    curr_change[current_delay] = c_epoch

                                    converged = True
                                    eval_mean_performance = eval_performance(network, inter_time, mem_delay, reset_input, seed)
                                    
                    else:
                        if np.all(np.average(result_buffer, axis=1) > stopping_crit):
                            if check_performance(network, inter_time, mem_delay, reset_input):
                                c_epoch = trial

                                converged = True
                                eval_mean_performance = eval_performance(network, inter_time, mem_delay, reset_input, seed)


                else:  # Mark trial type as failed
                    result_buffer[tmp_tp, int(result_buffer_idces[tmp_tp])] = 0
                    
                    if cur_on:
                        new_curriculum_buffer[tmp_tp, int(new_curriculum_buffer_idces[tmp_tp])] = 0


                # Increase (circular) buffer index
                result_buffer_idces[tmp_tp] += 1
                result_buffer_idces[tmp_tp] %= stopping_crit_window

                if cur_on:
                    new_curriculum_buffer_idces[tmp_tp] += 1
                    new_curriculum_buffer_idces[tmp_tp] %= stopping_crit_window

                # Compute average performance on each trial type
                for i in range(n_trial_types):
                    result_buffer_avgs[i, trial] = np.mean(result_buffer[i, :])
                    
                    if cur_on:
                        new_curriculum_buffer_avgs[i, trial] = np.mean(new_curriculum_buffer[i, :])


            # Obtain new task state, based on last network action
            new_input, reward, reset = task.do_step(action)


    # Done running experiment.
    # Compute performance averaged over all trials
    mean_performance = np.mean(trial_res[:c_epoch])

    if int(converged) == 0:
        eval_mean_performance = 0

    summary = {'unstable': int(unstable), 'convergence': int(converged), 'c_epoch': c_epoch,
               'mean_performance': mean_performance, 'eval_mean_performance': eval_mean_performance, 
               'learn_r': learn_r, 'seed': seed, 'explore_r': explore_r, 
               'decay': decay, 'n_mem': n_mem, 'beta_gate': float(learn_r * learn_g), 
               'theta': theta, 'sig_slope': sig_slope, 'int_trial':inter_time, 'mem_delay': mem_delay, 
               'reset_input': reset_input, 'curriculum_used': cur_on}

    # Save trained network
    network.save_network('ggsa_%i.cpickle' % (seed))


    if PLOTTING:
        # Show the graph, blocking execution
        print("\nClose graph to terminate.")

        # Update graphs:
        g.update_network_performance(trial_res)

        # Plot performance on all trial types separately
        for i in range(n_trial_types):
            y = result_buffer_avgs[i, :]
            g.update_subtask_performance(i, y)

        g.plot_convergence_trial(c_epoch, rescale=True)
        
        if cur_on:
            g.plot_curriculum_changes(curr_change.values())
        
        g.draw()
        plt.ioff()
        plt.legend(['left_pro', 'right_pro', 'left_anti', 'right_anti'],fontsize="small", loc='upper left')
        plt.savefig("ggsa_%i.jpeg" % (seed))
        plt.show(block=True)


    return summary



def eval_performance(network, inter_time, mem_delay, reset_input, seed):
    # Toggle learning and exploration off
    network.set_learning('off')
    network.set_exploration('off')

    # Number of evaluation trials to run
    eval_trials = 500
    perf_crit_window = 100

    eval_task = GGSA(inter_dur=inter_time, memory_dur=mem_delay)

    # Buffers for storing performance for all trial_types
    n_trial_types = len(eval_task.trial_types)
    eval_result_buffer = np.zeros((n_trial_types, perf_crit_window))
    eval_result_buffer_idces = np.zeros(n_trial_types)
    eval_result_buffer_avgs = np.zeros((n_trial_types, eval_trials))

    # Trial results (0: failure, 1: success)
    eval_trial_res = np.zeros(eval_trials)

    # Get size of task output (to determine input layer size)
    new_input = eval_task.display.to_output() 

    reward = 0.
    reset = False

    for eval_trial in range(0, eval_trials):

        trial_running = True

        while trial_running:

            # Get action from network based on latest output from Task
            if reset_input:
                action = network.do_step(np.hstack((new_input, int(reset))), reward)
            else:
                action = network.do_step(new_input, reward)

            if reset:  # End of trial detected:
                logging.getLogger(__name__).info("Trial {:d} end".format(eval_trial))

                trial_running = False
                tmp_tp = eval_task.last_trial_type

                if reward == eval_task.fin_reward:  # Mark trial as successful
                    logging.getLogger(__name__).info("Obtained reward!")

                    eval_trial_res[eval_trial] = 1

                    # Add result to results for specific input-pattern:
                    eval_result_buffer[tmp_tp, int(eval_result_buffer_idces[tmp_tp])] = 1

                else:  # Mark trial type as failed
                    eval_result_buffer[tmp_tp, int(eval_result_buffer_idces[tmp_tp])] = 0

                # Increase (circular) buffer index
                eval_result_buffer_idces[tmp_tp] += 1
                eval_result_buffer_idces[tmp_tp] %= perf_crit_window

                # Compute average performance on each trial type
                for i in range(n_trial_types):
                    eval_result_buffer_avgs[i, eval_trial] = np.mean(eval_result_buffer[i, :])

            # Obtain new task state, based on last network action
            new_input, reward, reset = eval_task.do_step(action)

    # Done running evaluation
    # Compute performance averaged over all trials
    eval_mean_performance = np.mean(eval_trial_res)

    network.set_learning('on')
    network.set_exploration('on')

    return eval_mean_performance


def check_performance(network, inter_time, mem_delay, reset_input):
    """
    Run all trial types without learning and exploration.
    Returns True if all valid input patterns are correctly dealt with.
    """

    # Toggle learning and exploration off
    network.set_learning('off')
    network.set_exploration('off')
    success = True

    # Iterate over trial types:
    task = GGSA(inter_dur=inter_time, memory_dur=mem_delay)

    for i in range(0, len(task.trial_types)):

        tmp_task = GGSA(inter_dur=inter_time, memory_dur=mem_delay)

        new_input = tmp_task.display.to_output()
        tmp_task.set_trial_type(i)

        reward = 0
        reset = False

        # Run trial type until completion
        while True:
            if reset_input:
                action = network.do_step(np.hstack((new_input, int(reset))), reward)
            else:
                action = network.do_step(new_input, reward)

            if reset:  # End of trial detected:
                # Check for failure
                if reward != tmp_task.fin_reward:
                    success = False
                break

            (new_input, reward, reset) = tmp_task.do_step(action)

        if not success:
            break

    # Reactivate Network
    network.set_learning('on')
    network.set_exploration('on')

    return success


def check_stable(vals):
    """
    Check whether any value in vals is NaN or extremely large
    """
    if np.max(np.abs(vals)) > 1e+3:
        return False
    if np.any(np.isnan(vals)):
        return False
    if np.any(np.isinf(vals)):
        return False
    return True


class Graphs(object):
    """
    Simple container class for plots
    """

    def __init__(self, max_trials, n_subtasks):
        """
        Create empty graphs with subplots for global network performance
        and detailed subtask performance
        """
        self.figure, self.axarr = plt.subplots(1, 2, sharey=True)

        self.axarr[0].set_title('Network performance')
        self.axarr[0].set_xlabel('Trials')
        self.axarr[0].set_ylabel('Performance')
        self.axarr[0].set_ylim([0, 1.05])
        self.filter_size = 100 

        self.max_trials = max_trials
        self.perf_line, = self.axarr[0].plot(list(range(max_trials)), np.zeros(self.max_trials, ), lw=2)

        self.axarr[1].set_title('Trial type performance')
        self.axarr[1].set_xlabel('Trials')
        self.perf_lines = self.axarr[1].plot(np.zeros((self.max_trials, n_subtasks)), lw=2)

        # callback for getting coordinate data from plots
        self.figure.canvas.mpl_connect('button_press_event', self.on_click)

        # Rotate x-labels
        for ax in self.axarr:
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=45)

        self.c_epoch = -1

    def on_click(self, event):
        """
        Callback for graphs
        """
        if event.button == 1:
            print("x={:2f}, y={:2f}".format(event.xdata, event.ydata))

    def update_network_performance(self, trial_res):
        """
        Plot smoothed network performance
        """
        y = np.convolve(np.ones(self.filter_size) / float(self.filter_size), trial_res, mode='same')
        self.perf_line.set_ydata(y)

    def update_subtask_performance(self, task_id, ydata):
        self.perf_lines[task_id].set_ydata(ydata)

    def plot_convergence_trial(self, c_epoch, rescale=False):
        self.c_epoch = c_epoch
        [ax.axvline(x=self.c_epoch, color='r', linestyle='--') for ax in self.axarr]

        # Rescale plots to learning time?
        if rescale and self.c_epoch != -1:
            [ax.set_xlim([0, self.c_epoch * 1.01]) for ax in self.axarr]

    def plot_curriculum_changes(self, curriculum_changes):
        for value in curriculum_changes:
            [ax.axvline(x=value, color ='gray',linestyle ="--") for ax in self.axarr]

    def draw(self):
        self.figure.canvas.draw()


def main():
    # Change the logging level for more/less detailed information
    # e.g. use level=logging.ERROR to see only messages logged to ERROR level (and higher)
    logging.basicConfig(format=' [%(levelname)s] %(name)s %(message)s', level=None)
    
    seed = 1
    example = run_experiment(seed)

    # Save results
    example_df = pd.DataFrame.from_dict(example, orient='index')
    example_df.to_csv('ggsa_%i.csv' % seed) 

    print(example_df)


if __name__ == '__main__':
    main()
