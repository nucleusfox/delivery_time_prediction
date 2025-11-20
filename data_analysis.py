import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# ---------------------------------------------------------
# Load delivery data
# ---------------------------------------------------------
def load_delivery_data(path="./data/delivery_data_rush_noise.csv"):
    df = pd.read_csv(path)
    
    # Add categorical columns
    df["is_rush_hour"] = df.apply(
        lambda r: int(
            (r["delivery_on_weekend"] == 0) and 
            (8 <= r["delivery_hour"] <= 11 or 16 <= r["delivery_hour"] <= 20)
        ),
        axis=1
    )
    df["is_heavy"] = (df["package_weight_lb"] >= 20).astype(int)
    df["is_big_package"] = (df["num_items"] >= 5).astype(int)
    df["package_combo"] = (
        df["is_big_package"].astype(int).astype(str) + "_" +
        df["is_heavy"].astype(int).astype(str)
    )
    
    # Map to meaningful labels
    mapping = {
        "1_1": 3, #"Big & Heavy",
        "1_0": 2, #"Big & Light",
        "0_1": 1, #"Small & Heavy",
        "0_0": 0 #"Small & Light",
    }
    df["package_combo"] = df["package_combo"].map(mapping)
    return df

# ---------------------------------------------------------
# Plot delivery data
# ---------------------------------------------------------
def plot_delivery_data(df):
    """
    Generates a scatter plot of delivery data.

    This function plots delivery time vs. distance, with markers colored by the
    time of day and styled (filled/hollow) based on whether the delivery
    was on a weekend.

    Args:
        df: A DataFrame containing required columns:
            'distance_miles', 'delivery_time_minutes',
            'delivery_hour', 'delivery_on_weekend'
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot setup ---
    plt.figure(figsize=(12, 7))
    ax = plt.axes()

    ax.set_facecolor('#d3ddeb') # <-- light gray blue

    # Text color
    text_color = 'black'
    grid_color = '#FFFFFF'
    legend_marker_color = '#444444'

    # Split weekday / weekend
    df_weekend = df[df['delivery_on_weekend'] == 1]
    df_weekday = df[df['delivery_on_weekend'] == 0]

    # Time-of-day colormap
    cmap = plt.get_cmap('YlOrRd')
    norm = Normalize(
        vmin=df['delivery_hour'].min(),
        vmax=df['delivery_hour'].max()
    )

    # --- Weekday (filled) ---
    scatter_weekday = plt.scatter(
        df_weekday['distance_miles'],
        df_weekday['delivery_time_min'],
        s=80,
        c=df_weekday['delivery_hour'],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        label='Weekday'
    )

    # --- Weekend (hollow) ---
    scatter_weekend = plt.scatter(
        df_weekend['distance_miles'],
        df_weekend['delivery_time_min'],
        s=80,
        facecolors='none',
        edgecolors=cmap(norm(df_weekend['delivery_hour'])),
        linewidths=1.5,
        alpha=0.9,
        label='Weekend'
    )

    # --- Fonts ---
    title_font = {'family': 'sans-serif', 'color': text_color,
                  'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color,
                  'size': 12}

    ax.set_title('Delivery Time vs. Distance', fontdict=title_font)
    ax.set_xlabel('Distance (miles)', fontdict=label_font)
    ax.set_ylabel('Delivery Time (minutes)', fontdict=label_font)

    # Ticks + spines coloring
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    # Grid styling
    ax.grid(True, which='both', linestyle='--',
            linewidth=0.6, color=grid_color, alpha=0.5)

    # --- Custom legend handles ---
    weekday_handle = Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor=legend_marker_color,
        markeredgecolor=legend_marker_color,
        markersize=8, linestyle='None',
        label="Weekday"
    )

    weekend_handle = Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor='none',
        markeredgecolor=legend_marker_color,
        markeredgewidth=1.5,
        markersize=8, linestyle='None',
        label="Weekend"
    )

    legend = ax.legend(
        handles=[weekday_handle, weekend_handle],
        labels=['Weekday', 'Weekend'],
        title='Day Type'
    )

    # Legend text styling
    plt.setp(legend.get_texts(), color=text_color, fontsize=12)
    plt.setp(legend.get_title(), color=text_color, fontsize=13, weight='bold')

    # --- Colorbar for time of day ---
    cbar = plt.colorbar(scatter_weekday, ax=ax)
    cbar.set_label('Time of Day (Hour)', fontdict=label_font)
    cbar.ax.tick_params(colors=text_color)

    plt.show()

# ---------------------------------------------------------
# Plot distance vs time, split by:
#    - weekend vs weekday
#    - rush hour vs non-rush hour
#    - heavy package vs light package
# ---------------------------------------------------------
def plot_distance_time_weekday_rush_heavy(distance, time, weekend, rush_hour, heavy_package, predicted_time=None, label=None):
    sorted_indices = torch.argsort(distance)
    
    distance = distance[sorted_indices]
    time = time[sorted_indices]
    weekend = weekend[sorted_indices]
    rush_hour = rush_hour[sorted_indices]
    heavy_package = heavy_package[sorted_indices]
    predicted_time = predicted_time[sorted_indices] if predicted_time is not None else None
    

    dist = distance.numpy()
    t = time.numpy()
    weekend = weekend[:, 0].numpy().astype(bool)
    rush_hour = rush_hour[:, 0].numpy().astype(bool)
    heavy_package = heavy_package[:, 0].numpy().astype(bool)
    pred = predicted_time.numpy() if predicted_time is not None else None


    # --- Plot --- setup
    plt.figure(figsize=(10, 6))

    # masks
    mask_weekday = (weekend == 0)
    mask_weekend = (weekend == 1)

    mask_rush = (rush_hour == 1)
    mask_nonrush = (rush_hour == 0)
    
    mask_heavy = (heavy_package == 1)
    mask_light = (heavy_package == 0)

    # WEEKDAY
    plt.scatter(dist[mask_weekday & mask_nonrush & mask_heavy],
                t[mask_weekday & mask_nonrush & mask_heavy],
                color="purple", alpha=0.6, marker="o", label="Weekday - Non-Rush - Heavy package")
    
    
    plt.scatter(dist[mask_weekday & mask_rush & mask_heavy],
                t[mask_weekday & mask_rush & mask_heavy],
                color="purple", alpha=0.9, marker="x", label="Weekday - Rush - Heavy package")

    plt.scatter(dist[mask_weekday & mask_nonrush & mask_light],
                t[mask_weekday & mask_nonrush & mask_light],
                color="blue", alpha=0.6, marker="o", label="Weekday - Non-Rush - Light package")
    
    
    plt.scatter(dist[mask_weekday & mask_rush & mask_light],
                t[mask_weekday & mask_rush & mask_light],
                color="blue", alpha=0.9, marker="x", label="Weekday - Rush - Light package")

    # # WEEKEND
    plt.scatter(dist[mask_weekend & mask_nonrush & mask_heavy],
                t[mask_weekend & mask_nonrush & mask_heavy],
                color="red", alpha=0.6, marker="o", label="Weekend - Non-Rush - Heavy package")
    
    plt.scatter(dist[mask_weekend & mask_nonrush & mask_light],
                t[mask_weekend & mask_nonrush & mask_light],
                color="orange", alpha=0.6, marker="o", label="Weekend - Non-Rush - Light package")

    # Optional predicted points
    if pred is not None:
        if label is not None:
            label = "Predictions for " + label
        else:
            label = "Predicted Time"
        plt.plot(dist, pred, color="green", alpha=0.3, marker=".", label=label)

    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time vs Distance (Weekday/Weekend + Rush Hour + Heavy/Light package)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predicted_time_rush_light(results_dict, predicted_point):
    # workday & rush & light
    features = results_dict['features']
    combo_mask = (features[:, 4] == 0) & (features[:, 5] == 1) & (features[:, 1] == 0)
            
    distances = results_dict['raw_distances'][combo_mask]
    time=results_dict['raw_targets'][combo_mask]
    weekend=results_dict['weekends_col'][combo_mask]
    rush_hour=results_dict['rush_hour_col'][combo_mask]
    heavy_package=results_dict['heavy_package_col'][combo_mask]
    
    
    sorted_indices = torch.argsort(distances)
    
    distances = distances[sorted_indices]
    time = time[sorted_indices]
    weekend = weekend[sorted_indices]
    rush_hour = rush_hour[sorted_indices]
    heavy_package = heavy_package[sorted_indices]
    
    dist = distances.numpy()
    t = time.numpy()
    weekend = weekend[:, 0].numpy().astype(bool)
    rush_hour = rush_hour[:, 0].numpy().astype(bool)
    heavy_package = heavy_package[:, 0].numpy().astype(bool)


    # --- Plot --- setup
    plt.figure(figsize=(10, 6))

    # masks
    mask_weekday = (weekend == 0)
    mask_weekend = (weekend == 1)

    mask_rush = (rush_hour == 1)
    mask_nonrush = (rush_hour == 0)
    
    mask_heavy = (heavy_package == 1)
    mask_light = (heavy_package == 0)

    # WEEKDAY
    plt.scatter(dist[mask_weekday & mask_rush & mask_light],
                t[mask_weekday & mask_rush & mask_light],
                color="blue", alpha=0.9, marker="x", label="Weekday - Rush - Light package")


    # redicted point
    plt.scatter(predicted_point[0], predicted_point[1], color="red", alpha=0.9, marker="x", label="Predicted time")

    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time vs Distance (Weekday/Weekend + Rush Hour + Heavy/Light package)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_predicted_time_norush_heavy(results_dict, predicted_point):
    # workday & no rush & heavy
    features = results_dict['features']
    combo_mask = (features[:, 4] == 1) & (features[:, 5] == 0) & (features[:, 1] == 1)
            
    distances = results_dict['raw_distances'][combo_mask]
    time=results_dict['raw_targets'][combo_mask]
    weekend=results_dict['weekends_col'][combo_mask]
    rush_hour=results_dict['rush_hour_col'][combo_mask]
    heavy_package=results_dict['heavy_package_col'][combo_mask]
    
    
    sorted_indices = torch.argsort(distances)
    
    distances = distances[sorted_indices]
    time = time[sorted_indices]
    weekend = weekend[sorted_indices]
    rush_hour = rush_hour[sorted_indices]
    heavy_package = heavy_package[sorted_indices]
    
    dist = distances.numpy()
    t = time.numpy()
    weekend = weekend[:, 0].numpy().astype(bool)
    rush_hour = rush_hour[:, 0].numpy().astype(bool)
    heavy_package = heavy_package[:, 0].numpy().astype(bool)


    # --- Plot --- setup
    plt.figure(figsize=(10, 6))

    # masks
    mask_weekday = (weekend == 0)
    mask_weekend = (weekend == 1)

    mask_rush = (rush_hour == 1)
    mask_nonrush = (rush_hour == 0)
    
    mask_heavy = (heavy_package == 1)
    mask_light = (heavy_package == 0)


    # # WEEKEND
    plt.scatter(dist[mask_weekend & mask_nonrush & mask_heavy],
                t[mask_weekend & mask_nonrush & mask_heavy],
                color="orange", alpha=0.6, marker="o", label="Weekend - Non-Rush - Heavy package")

    # redicted point
    plt.scatter(predicted_point[0], predicted_point[1], color="red", alpha=0.9, marker="o", label="Predicted time")

    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time vs Distance (Weekday/Weekend + Rush Hour + Heavy/Light package)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
def plot_predicted_time_rush_heavy(results_dict, predicted_point):
    # workday & rush & heavy
    features = results_dict['features']
    combo_mask = (features[:, 4] == 0) & (features[:, 5] == 1) & (features[:, 1] == 1)
            
    distances = results_dict['raw_distances'][combo_mask]
    time=results_dict['raw_targets'][combo_mask]
    weekend=results_dict['weekends_col'][combo_mask]
    rush_hour=results_dict['rush_hour_col'][combo_mask]
    heavy_package=results_dict['heavy_package_col'][combo_mask]
    
    
    sorted_indices = torch.argsort(distances)
    
    distances = distances[sorted_indices]
    time = time[sorted_indices]
    weekend = weekend[sorted_indices]
    rush_hour = rush_hour[sorted_indices]
    heavy_package = heavy_package[sorted_indices]
    
    dist = distances.numpy()
    t = time.numpy()
    weekend = weekend[:, 0].numpy().astype(bool)
    rush_hour = rush_hour[:, 0].numpy().astype(bool)
    heavy_package = heavy_package[:, 0].numpy().astype(bool)


    # --- Plot --- setup
    plt.figure(figsize=(10, 6))

    # masks
    mask_weekday = (weekend == 0)
    mask_weekend = (weekend == 1)

    mask_rush = (rush_hour == 1)
    mask_nonrush = (rush_hour == 0)
    
    mask_heavy = (heavy_package == 1)
    mask_light = (heavy_package == 0)

    # WEEKDAY
    plt.scatter(dist[mask_weekday & mask_rush & mask_heavy],
                t[mask_weekday & mask_rush & mask_heavy],
                color="purple", alpha=0.9, marker="x", label="Weekday - Rush - Heavy package")

    # redicted point
    plt.scatter(predicted_point[0], predicted_point[1], color="red", alpha=0.9, marker="x", label="Predicted time")

    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time vs Distance (Weekday/Weekend + Rush Hour + Heavy/Light package)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
# ---------------------------------------------------------
# Plot losses
# ---------------------------------------------------------
def plot_losses(losses, step=100):
    epochs = [i * step for i in range(len(losses))]

    plt.figure(figsize=(10,6))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.show()

# ---------------------------------------------------------
# Plot predictions vs truth
# ---------------------------------------------------------
def plot_prediction_vs_target(predicted, target):
    predicted = predicted.numpy()
    target = target.numpy()

    text_color = 'black'
    grid_color = '#FFFFFF'
    perfect_prediction_line_color = '#E64C29'

    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    
    plt.scatter(predicted,
                target,
                color="blue", alpha=0.6, marker="o", label="Predicted vs Target")

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(
        lims, lims,
        linestyle='-',
        color=perfect_prediction_line_color,
        alpha=0.75,
        zorder=0,
        label='Perfect Prediction'
    )

    # Define font properties
    title_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'normal', 'size': 12}

    # Set titles and labels
    ax.set_title('Actual vs. Predicted Delivery Times', fontdict=title_font)
    ax.set_xlabel('Actual Delivery Time (minutes)', fontdict=label_font)
    ax.set_ylabel('Predicted Delivery Time (minutes)', fontdict=label_font)

    # Style ticks and spines
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    # Style grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)
    plt.xlabel("Predicted delivery time")
    plt.ylabel("Recorded delivery time")
    plt.title("Actual vs Predicted Delivery Time")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# STATISTICS TABLES
# ---------------------------------------------------------------------------
def summary_stats(df):
    """Basic descriptive statistics."""
    return df.describe()


def weekend_vs_weekday_stats(df):
    """Compare delivery time by weekend/weekday."""
    return df.groupby("delivery_on_weekend")["delivery_time_min"].agg(
        ["count", "mean", "std", "min", "max"]
    ).rename(index={0: "weekday", 1: "weekend"})


def tip_effect_stats(df):
    """Impact of tip on delivery time."""
    return df.groupby("tip_given")["delivery_time_min"].agg(
        ["count", "mean", "std", "min", "max"]
    ).rename(index={0: "no_tip", 1: "tip"})


def rush_hour_effect_stats(df):
    """Effect of rush hour on weekdays."""
    return df.groupby("is_rush_hour")["delivery_time_min"].agg(
        ["count", "mean", "std", "min", "max"]
    ).rename(index={0: "not_rush", 1: "rush"})


def weight_group_stats(df):
    """Light (<20) vs Heavy (>=20)."""
    return df.groupby("is_heavy")["delivery_time_min"].agg(
        ["count", "mean", "std", "min", "max"]
    ).rename(index={0: "light", 1: "heavy"})


def correlation_table(df):
    """Pearson correlations between numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number])
    return numeric_cols.corr()


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------
# TIME vs DISTANCE (multiple splits)
def plot_time_vs_distance_weekend(df):
    plt.figure(figsize=(10,7))
    sns.scatterplot(
        data=df,
        x="distance_miles",
        y="delivery_time_min",
        hue="delivery_on_weekend",
        style="is_rush_hour",
        palette="Set2",
        s=40
    )
    plt.title("Delivery Time vs Distance (Weekend vs Weekday, Rush Hour Split)")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.legend(title="0=weekday / 1=weekend")
    plt.show()


def plot_time_vs_distance_tip(df):
    plt.figure(figsize=(10,7))
    sns.scatterplot(
        data=df,
        x="distance_miles",
        y="delivery_time_min",
        hue="tip_given",
        palette="coolwarm",
        s=40
    )
    plt.title("Delivery Time vs Distance (Tip vs No Tip)")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.legend(title="Tip Given")
    plt.show()


def plot_time_vs_distance_weight(df):
    plt.figure(figsize=(10,7))
    sns.scatterplot(
        data=df,
        x="distance_miles",
        y="delivery_time_min",
        hue="is_heavy",
        palette="viridis",
        s=40
    )
    plt.title("Delivery Time vs Distance (Light vs Heavy Packages)")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.legend(title="Heavy (>=20lb)")
    plt.show()

# -----------------------------
# DISTRIBUTIONS
def plot_time_distribution(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df["delivery_time_min"], bins=40, kde=True)
    plt.title("Distribution of Delivery Times")
    plt.xlabel("Delivery Time (minutes)")
    plt.ylabel("Count")
    plt.show()


def plot_weight_distribution(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df["package_weight_lb"], bins=30, kde=True)
    plt.title("Distribution of Package Weights")
    plt.xlabel("Package Weight (lb)")
    plt.ylabel("Count")
    plt.show()


# -----------------------------
# CATEGORY COMPARISON PLOTS
def boxplot_time_by_weekend(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="delivery_on_weekend", y="delivery_time_min", palette="Set3")
    plt.xlabel("Weekend (1=yes)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time Comparison: Weekend vs Weekday")
    plt.show()


def boxplot_time_by_rushhour(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="is_rush_hour", y="delivery_time_min", palette="Set2")
    plt.xlabel("Rush Hour (1=yes)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time Comparison: Rush Hour vs Non-Rush Hour")
    plt.show()


def boxplot_time_by_tip(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="tip_given", y="delivery_time_min", palette="coolwarm")
    plt.xlabel("Tip Given (1=yes)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time Comparison: Tip vs No Tip")
    plt.show()


def boxplot_time_by_weight(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="is_heavy", y="delivery_time_min", palette="viridis")
    plt.xlabel("Heavy Package (>=20lb)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time by Weight Group")
    plt.show()

def boxplot_time_by_number(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="is_big_package", y="delivery_time_min", palette="viridis")
    plt.xlabel("Big Packages (>5 items)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time by Number of items Group")
    plt.show()

def boxplot_time_by_package_combo(df):
    mapping = {0: "Small & Light", 1: "Small & Heavy", 2: "Big & Light", 3: "Big & Heavy"}
    
    plt.figure(figsize=(10,6))
    ax = sns.boxplot(data=df, x="package_combo", y="delivery_time_min", palette="viridis")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([mapping[i] for i in [0, 1, 2, 3]])
    plt.xlabel("Big Packages (>5 items)")
    plt.ylabel("Delivery Time (min)")
    plt.title("Delivery Time by Number of items Group")
    plt.show()
    
# -----------------------------
# HEATMAPS & LINEAR RELATIONSHIPS
def correlation_heatmap(df):
    plt.figure(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


def plot_distance_time_regression(df):
    plt.figure(figsize=(10,6))
    sns.regplot(
        data=df,
        x="distance_miles",
        y="delivery_time_min",
        scatter_kws={'s':20, 'alpha':0.6},
        line_kws={'lw':2}
    )
    plt.title("Linear Relationship: Delivery Time vs Distance")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Delivery Time (min)")
    plt.show()

# ---------------------------------------------------------------------------
# EXPORT FUNCTIONS
# ---------------------------------------------------------------------------
def export_table(df, name="table.csv"):
    df.to_csv(name, index=True)
    print(f"Saved table → {name}")