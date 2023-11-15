import pandas as pd
import graphistry
import os

def creat_network():
    graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="yang_li", password="Aa239102")
    file_path = os.path.join('uploads', 'resultFile.csv')
    try:
        adjacency_matrix = pd.read_csv(file_path, index_col=0, skipinitialspace=True)

    except FileNotFoundError:
        return f"错误：未找到文件'{file_path}'。请确保文件存在。"

    if adjacency_matrix is not None:
        print(adjacency_matrix)
        nodes = adjacency_matrix.index.tolist()
        edges = []
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix.iloc[i])):
                if adjacency_matrix.iloc[i, j]:
                    edges.append((nodes[i], nodes[j]))


        df = pd.DataFrame(edges, columns=["Source", "Target"])
        g = graphistry.bind(source='Source', destination='Target')
        g = g.edges(df)
        g.plot()

if __name__ == '__main__':
    creat_network()

