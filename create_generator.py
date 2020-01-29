import numpy as np
import ssl
from warnings import warn
import zipfile
import os
import LFPy
import neuron
from neuron import h


def generator(spike_times,xs,ys,zs,sigma):
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
    
    # define parameters for extracellular recording electrodes
    electrodeParameters = {
        'sigma' : sigma,                  # extracellular conductivity
        'x' : cell.xmid+xs,      
        'y' : cell.ymid+ys,
        'z' : cell.zmid+zs,
        'method' : 'soma_as_point',     #sphere source soma segment
        'N' : np.array([[0, 1, 0]]*cell.xmid.size), #surface normals
        'r' : 2.5,                      # contact site radius
        'n' : 20,                       # datapoints for averaging
    }

    # create extracellular electrode object for LFPs on grid in xz-plane
    electrode = LFPy.RecExtElectrode(**electrodeParameters)

    # simulate generator cell
    cell.simulate(electrode = electrode,rec_vmem=True)

    return electrode,synapse.i,cell.vmem,cell.somav
