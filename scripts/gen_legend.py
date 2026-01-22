import pylab
import matplotlib.pyplot as plt

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'


color_mapping = {"PV": "#1f77b4", 
                 "P": "#ff0318",
                 "V": "#ff7f03",
                 "None": "#2ca02c",
                 "AR": "#1f77b4",
                 "TEA": "#9467bd"}

def export_legend(ax, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=10,)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches = bbox)


# legend_labels = ("PV", "P", "V", "None")
legend_labels = ("AR", "TEA")

fig = plt.figure()
ax = fig.add_subplot(111)
for legend_text in legend_labels:
    ax.plot(range(10), pylab.randn(10), label=legend_text, color=color_mapping[legend_text])
fig.show()
# fig.savefig("demo.pdf")
export_legend(ax, "legend_2.pdf")