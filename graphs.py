import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(df):
    metrics = df.columns[2:]
    print(metrics)
    n = len(metrics)
    plt.figure(figsize=(14, 8))

    for metric in metrics:
        print(metric)
        plt.plot(df.index, df[metric], marker='o', label=metric)

    plt.title('Model Performance Metrics')
    plt.xlabel('Samples')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('files/segnet_graph.png',dpi=300)
    plt.show()

def replace_zeros(df):
    for col in df.columns:
        for i in range(len(df)):
            if df.at[i, col] == 0:
                above = df.at[i-1, col] if i > 0 else None
                below = df.at[i+1, col] if i < len(df) - 1 else None
                
                if above is not None and below is not None:
                    df.at[i, col] = (above + below) / 2
                elif above is not None:
                    df.at[i, col] = above
                elif below is not None:
                    df.at[i, col] = below
    
    return df

df = pd.read_csv('files/segnet_data_train.csv')
df = replace_zeros(df)

plot_metrics(df)

