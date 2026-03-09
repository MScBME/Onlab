import pandas as pd

def export_positions(xs, ts, path):

    df = pd.DataFrame({
        "time": ts,
        "x_position": xs
    })

    df.to_csv(path, index=False)