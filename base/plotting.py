import numpy as np
import matplotlib.pyplot as plt


def plotting_fun(states, inputs, outputs, data, var=[], costs=[], demand = []):
    if var[0] == 'BO':

        BO = []
        for t in states.keys():
            BO.append([states[t]['backorder'][i] for i in states[t]['backorder']])

        plt.figure(figsize = (3,2))
        plt.step([k for k in states.keys()], np.array(BO))
        plt.legend([i for i in states[t]['backorder']], fontsize = 8)
        plt.title('Backorder')
        plt.tight_layout()
        plt.draw()

    if var[0] == 'omega':
        waste = []
        for t in outputs.keys():
            waste.append([outputs[t]['omega'][i] for i in outputs[t]['omega']])

        plt.figure(figsize=(3, 2))
        plt.step([k for k in outputs.keys()], np.array(waste))
        plt.legend([i for i in outputs[t]['omega']], fontsize=8)
        plt.title('Waste')
        plt.tight_layout()
        plt.draw()


    if var[0] == 'I':

        I = []
        for t in states.keys():
            I.append([sum(sum(states[t]['inventory'][(i, q, k)]
                              for k in range(1, data['temperature'] + 1))
                          for q in range(1, data['quality'] + 1))
                      for i in var[1]])



        plt.figure(figsize = (3,2))
        plt.step([t for t in states.keys()], np.array(I))
        plt.legend(var[1], fontsize = 8, ncol = 2)
        plt.title('Inventory')
        plt.tight_layout()
        plt.draw()

    if var[0] == 's':
        s, d = [], []
        for t in inputs.keys():
            s.append([sum(inputs[t]['r'][(i, q)] for q in range(1, data['quality'] + 1)) for i in var[1]])
            d.append([demand[(i,t)] for i in var[1]])

        plt.figure(figsize = (3,2))
        plt.step([t for t in inputs.keys()], np.array(s))
        # plt.step([t for t in inputs.keys()], np.array(d), linestyle = '-.')
        plt.legend(var[1], fontsize = 8)
        plt.title('Sales')

    if var[0] == 'y':
        y = []
        for t in inputs.keys():
            y.append([sum(inputs[t]['y'][(i, q)] for q in range(1, data['quality'] + 1)) for i in var[1]])

        plt.figure(figsize = (3,2))
        plt.step([t for t in inputs.keys()], np.array(s))
        plt.legend(var[1], fontsize = 8)
        plt.title('Production')
        plt.show()

    if var[0] == 'x':
        x = []
        for t in inputs.keys():
            x.append([sum(sum(inputs[t]['x'][(i, j, q, k)] for q in range(1, data['quality'] + 1))
                          for k in range(1, data['temperature'] + 1))
                      for i, j in var[1]])

        plt.figure(figsize = (3,2))
        plt.step([t for t in inputs.keys()], np.array(x))
        plt.legend(var[1], fontsize = 8, ncol = 2)
        plt.title('Shipment')
        plt.tight_layout()
        plt.draw()

    if var[0] == 'costs':
        plt.figure(figsize = (3,2))
        plt.plot(costs)
        # plt.axhline(y=(20*5.8000000000e+02), color = 'k', linestyle = '--')
        plt.title('Costs')
        plt.tight_layout()
        plt.draw()


def plotting_fun_est(est_states, true_states, var=[], costs=[]):
    if var[0] == 'BO':

        BO = []
        for t in est_states.keys():
            BO.append([est_states[t]['backorder'][i] for i in est_states[t]['backorder']])

        plt.figure(figsize=(3, 2))
        plt.step([k for k in est_states.keys()], np.array(BO))
        plt.legend([i for i in est_states[t]['backorder']], fontsize = 8)

        plt.title('Backorder')

        BO = []
        for t in true_states.keys():
            BO.append([true_states[t]['backorder'][i] for i in true_states[t]['backorder']])

        plt.step([k for k in true_states.keys()], np.array(BO), color = 'k', linestyle = '--')
        plt.tight_layout()
        plt.draw()