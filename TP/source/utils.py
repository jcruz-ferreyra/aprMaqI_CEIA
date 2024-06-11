import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def phi_coefficient(confusion_matrix):
    chi2, p, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi = np.sqrt(chi2 / n)
    return phi, p


def evaluate_predictions(y_true, y_pred, figsize=(3, 3)):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Mapping of labels
    labels = ["No", "Yes"]

    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted Values")
    plt.ylabel("Real Values")
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate evaluation metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")


def print_missing_perc(df, column):
    missing_perc = round((df[column].isna().sum() / df.shape[0]) * 100, 1)
    print(f"Porcentaje de valores faltantes en la columna {column}: {missing_perc}%")


def plot_locations_over_time(df, color="blue"):
    locations = df["Location"].unique()

    _, ax = plt.subplots(figsize=(12, 7))

    # Plot data for each location
    for location in locations:
        filt = df["Location"] == location
        df_location = df.loc[filt, :]

        # Plot observations present in dataframe
        ax.plot(
            df_location["Date"],
            [location] * len(df_location),
            "o-",
            color=color,
            label=location,
            markersize=0.5,
            linewidth=0.05,
        )

        # Plot null values in target column
        null_indices = df_location.loc[df_location["RainTomorrow"].isnull()].index
        for idx in null_indices:
            ax.plot(df_location.loc[idx, "Date"], location, "ko", markersize=0.15)

    # Customize the plot
    ax.set_yticks(np.arange(len(locations)))
    ax.set_yticklabels(
        locations, fontsize="x-small"
    )  # Increase fontsize for y-axis labels
    ax.set_ylim(-0.5, len(locations) - 0.5)

    xticks = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="6MS")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.strftime("%Y-%m"), fontsize="x-small", rotation=90)

    ax.grid(True, linestyle=":", alpha=0.5)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightgreen",
            markersize=5,
            label="Observación presente en el dataframe",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=5,
            label="Observación con valor ausente en la columna 'RainTomorrow'",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=2,
        fontsize=7,
    )

    plt.suptitle(
        "Observaciones presentes en la serie temporal por centro meteorológico",
        fontsize=10,
    )

    plt.show()
