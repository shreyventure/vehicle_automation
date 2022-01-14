import pandas as pd
def getColors(lables) -> list:
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'grey']
    lables_uni = lables.unique()
    d = {}
    i = 0
    for ele in lables_uni:
        d[ele] = colors[i]
        i += 1

    ret_colors = []
    for ele in lables:
        ret_colors.append(d[ele])

    return ret_colors