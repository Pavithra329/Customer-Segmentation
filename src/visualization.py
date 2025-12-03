import plotly.express as px
import pandas as pd

def plot_3d_clusters(data, labels, title="3D Clusters"):
    df = pd.DataFrame(data, columns=['PC1','PC2','PC3'])
    df['Cluster'] = labels.astype(str)
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
                        color='Cluster', opacity=0.7)
    fig.update_layout(title=title)
    fig.show()
