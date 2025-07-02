import matplotlib.pyplot as plt

def plot_feature_importance():
    fig, ax = plt.subplots()
    ax.bar(["Temp", "Rain", "Humidity"], [0.3, 0.4, 0.3])
    ax.set_title("Feature Importance (Example)")
    return fig