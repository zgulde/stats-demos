import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

import sklearn.preprocessing

SCALERS = {
    "min-max": sklearn.preprocessing.MinMaxScaler(),
    "standard (z-score)": sklearn.preprocessing.StandardScaler(),
    "quantile (uniform)": sklearn.preprocessing.QuantileTransformer(),
    "quantile (normal)": sklearn.preprocessing.QuantileTransformer(
        output_distribution="normal"
    ),
    "power (yeo-johnson)": sklearn.preprocessing.PowerTransformer(),
    "power (box-cox)": sklearn.preprocessing.PowerTransformer(method="box-cox"),
    "robust": sklearn.preprocessing.RobustScaler(),
}

DATASETS = {
    "small": np.array([0, 1, 2, 3, 100]),
    "uniform random": np.random.uniform(1, 100, 1000),
    "skewed left": stats.skewnorm(-5, 50, 5).rvs(1000),
    "skewed right": stats.skewnorm(5, 50, 5).rvs(1000),
    "normal": np.random.normal(50, 5, 1000),
}


def get_scaler_names():
    return SCALERS.keys()


def get_dataset_names():
    return DATASETS.keys()


def visualize_scaler(scaler_name, data_name, base=6):
    scaler = SCALERS[scaler_name]  # todo: validity check
    data = DATASETS[data_name]  # todo: validity check
    return _visualize_scaler(scaler, data, base, scaler_name, data_name)


def _visualize_scaler(scaler, data, base, scaler_name, data_name):
    nrows = 2
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * base, nrows * base))

    df = pd.DataFrame(dict(original=data))
    df["scaled"] = scaler.fit_transform(df[["original"]])

    # scatter plot
    df.plot.scatter(
        y="scaled", x="original", ax=axs[0, 0], title="Orignial vs Scaled Value"
    )

    # table
    sample = (
        df.set_index(pd.Index([""] * df.shape[0]))
        .sample(8 if df.shape[0] >= 8 else df.shape[0])
        .sort_values(by="original")
        .round(3)
    )
    pd.plotting.table(
        ax=axs[0][1], data=sample, loc="upper right", bbox=[0, 0, 1, 1], colLoc="right"
    )
    axs[0, 1].get_xaxis().set_visible(False)
    axs[0, 1].get_yaxis().set_visible(False)
    axs[0, 1].set(title="8 samples" if df.shape[0] >= 8 else "Data and Transformations")

    axs[1, 0].hist(df.original, bins=25)
    axs[1, 0].set(title="Histogram of Original Data")
    axs[1, 1].hist(df.scaled, bins=25)
    axs[1, 1].set(title="Histogram of Scaled Data")

    for ax in axs[1, :]:
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()

    return fig, axs
