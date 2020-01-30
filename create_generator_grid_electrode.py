import numpy as np
import ssl
from warnings import warn
import zipfile
import os
import LFPy
import neuron
from neuron import h


def generator_grid (spike_times,xs,ys,zs,sigma):
    # define cell parameters used as input to cell-class
    cellParameters = {
        'morphology' : 'morphology/mitral/mitral.hoc',     # mitral neuron (olfactory)
    #     'morphology' : 'morphology/mitral.hoc',            # mitral neuron (olfactory)
    #     'morphology' : 'morphology/purkinje.hoc',          # purkinje neuron (cerebellum)
    #     'morphology' : 'morphology/pyramidal_layer2.hoc',  # pyramidal neuron (cortex)
    #     'morphology' : 'morphology/pyramidal_layer5.hoc',  # pyramidal neuron (cortex)
    #     'morphology' : 'morphology/aspiny_layer3.hoc',     # aspiny neuron (cortex)
    #     'morphology' : 'morphology/stellate_layer4.hoc' ,   # stellate neuron (cortex)
        'cm' : 1.0,                 # membrane capacitance
        'Ra' : 150.,                # axial resistance
        'passive' : True,           # turn on NEURONs passive mechanism for all sections
        'nsegs_method' : None,      # spatial discretization method
        'dt' : 2**-6,               # simulation time step size
        'tstart' : 0,               # start time of simulation
        'tstop' : 40,               # stop simulation
        'v_init' : -60,             # initial crossmembrane potential
        'celsius': 34,
        'pt3d' : True,
        'extracellular': True,
        'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -60},
    }
    
    # delete old sections from NEURON namespace
    LFPy.cell.neuron.h("forall delete_section()")

    # initialize cell instance, using the LFPy.Cell class
    cell = LFPy.Cell(**cellParameters)
    cell.set_rotation(x=4.729, y=-3.166)
    cell.xmid += xs  
    print(cell.xmid[0])
    
    # create synapse
    synapse_parameters = {
        'idx' : cell.get_closest_idx(x=xs, y=ys, z=zs),  # place sinapse at soma coordinates
        'e' : 0,                     # reversal potential
        'syntype' : 'ExpSyn',        # synapse type
        'tau1' : 0.5,                # synaptic time constant
        'tau2' : 2,
        'weight' : 0.05,             # synaptic weight
        'record_current' : True,     # record synapse current
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array(spike_times))      # spike times
    
    # create a grid of measurement locations, in (mum)
    X, Z = np.mgrid[-600:601:10, -100:701:10]
    Y = np.zeros(X.shape)

    # define electrode parameters
    grid_electrode_parameters = {
        'sigma' : 0.3,             # extracellular conductivity
        'x' : X.flatten(),         # electrode requires 1d vector of positions
        'y' : Y.flatten(),
        'z' : Z.flatten(),
        'verbose': True,
    }
    # run simulation, electrode object argument in cell.simulate
    cell.simulate(rec_imem=True)
    
    # create electrode objects
    grid_electrode = LFPy.RecExtElectrode(cell,**grid_electrode_parameters)
    
    grid_electrode.calc_lfp()

    return grid_electrode.LFP, synapse.i, X, Y,Z
