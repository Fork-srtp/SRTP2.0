def print_graph_detail(graph):
    """
    格式化显示Graph参数
    :param graph:
    :return:
    """
    import networkx as nx
    dst = {"nodes"    : nx.number_of_nodes(graph),
           "edges"    : nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates" : nx.number_of_isolates(graph),
           "覆盖度"      : 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    print_table(dst)

def print_table(dst):
    table_title = list(dst.keys())
    from prettytable import PrettyTable
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    print(table)