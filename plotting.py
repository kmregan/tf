import plotly
import plotly.plotly as pltly
import plotly.graph_objs as go
import numpy as np

plotly.offline.init_notebook_mode()

def plot_2d_regression_results(data, w, b, title, offline=True):
    '''Return plotly graph object suitable for Jupyter notebooks that
       holds a scatter plot of data examples with a hyperplane built
       from the model (w,b). The hyperplane represents the 
       predictions of the model.
   
       We expect data to be an Mx3 matrix of M examples.'''

    x,y,z = data[:,0], data[:,1], data[:, 2]
    examples = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='green',
        )
    )
    x_space = np.linspace(min(x), max(x), 10)
    y_space = np.linspace(min(y), max(y), 10)
    x_grid, y_grid = np.meshgrid(x_space, y_space)
    z_space = w[0] * x_grid + w[1] * y_grid 

    prediction_hyperplane = go.Surface(
        x=x_space, y=y_space, z=z_space,
        opacity=0.99,
        colorscale='Greens',
        autocolorscale=False,
        showscale=False,
        zauto=False,
    )
    data = [examples, prediction_hyperplane]
    layout = go.Layout(
        title=title,
        margin=dict(
            l=0,
            r=0,
            b=0,
        )
    )
    fig = go.Figure(data=data, layout=layout)
    
    return plotly.offline.iplot(fig) if offline else pltly.plot(fig)
