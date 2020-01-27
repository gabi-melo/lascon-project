import matplotlib.pyplot as plt
import numpy as np

def plot_cell(cell, x_off=0, y_off=0, ax=None, voltage=None, vmin=None, vmax=None,
              palette='plasma', color='k', rescale=None):
    """
    Draws the cell morphology at the cells coordinates offseting by xmid and zmid
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.cm import get_cmap
    import numpy as np
    
    if ax is None:
        ax = plt.gca()
        
    if voltage is None:
        colors = color
    else:
        vmin = voltage.min() if vmin is None else vmin
        vmax = voltage.max() if vmax is None else vmax
        normalized_voltage = (voltage - vmin)/(vmax-vmin)
        colors = get_cmap(palette)(normalized_voltage)
       
    poly = cell.get_idx_polygons()
        
    if rescale is not None:
        raise NotImplementedError("rescaling has not been implemented yet")
#         xdist = (np.vstack([i for i,_ in poly]) - cell.xmid.reshape(81,1))
#         xres = cell.xmid.reshape(81,1) + rescale*xdist

#         ydist = (np.vstack([i for _,i in poly]) - cell.ymid.reshape(81,1))
#         yres = cell.ymid.reshape(81,1) + rescale*ydist

#         poly = list(zip(xres,yres))       
        
    zips = [list(zip(x + x_off, y+y_off)) for x, y in poly]
    drawing = PolyCollection(zips, edgecolors='none', facecolors = colors)
    
    ax.add_collection(drawing)
    
    axlim = np.array((ax.get_xlim(),ax.get_ylim())).T
    
    xmax, ymax = np.vstack((np.array(zips).max(0), axlim)).max(0)
    xmin, ymin = np.vstack((np.array(zips).min(0), axlim)).min(0)
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(ymin, ymax)
    return ax