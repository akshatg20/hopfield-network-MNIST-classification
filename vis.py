import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import patternLib

# function to plot a pattern
def see_pattern(pattern, ref = None, diff_code = 0):
    plt.figure()
    if ref is None:
        p = pattern
        overlap = 1
    else:
        p = patternLib.get_diff_pattern(pattern, ref, diff_code)
        overlap = patternLib.compute_overlap(pattern, ref)

    plt.imshow(p, interpolation="nearest", cmap='brg')
    if ref is not None:
        plt.title("m = {:0.2f}".format(round(overlap, 2)))
    plt.axis("off")
    plt.show()

# function to see a list of patterns
def see_pattern_list(pattern_list):
    f, ax = plt.subplots(1, len(pattern_list))
    title_pattern = "P{0}"
    if len(pattern_list) == 1:
        ax = [ax]
    
    for i in range(len(pattern_list)):
        p = pattern_list[i]
        if np.max(p) == np.min(p):
            ax[i].imshow(p, interpolation="nearest", cmap='RdYlBu')
        else:
            ax[i].imshow(p, interpolation="nearest", cmap='brg')

        ax[i].set_title(title_pattern.format(i))
        ax[i].axis("off")    

# function to plot the overlap
def see_overlap(overlap_matrix):
    plt.imshow(overlap_matrix, interpolation="nearest", cmap='bwr')
    plt.title("pattern overlap m(i,j)")
    plt.xlabel("pattern j")
    plt.ylabel("pattern i")
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='bwr'))
    plt.show()

def plot_state_sequence_and_overlap(state_sequence, pattern_list, ref_idx = None, suptitle=None, figsize = (10,6)):
    f, ax = plt.subplots(2, len(state_sequence), figsize = figsize)
    if len(state_sequence) == 1:
        ax = [ax]
    
    print()

    for i in range(len(state_sequence)):
        p = 0
        if ref_idx is None:
            p = state_sequence[i]
        else:
            ref = pattern_list[ref_idx]
            p = patternLib.get_diff_pattern(state_sequence[i], ref, diff_code=-0.2)
        if np.max(p) == np.min(p):
            ax[0,i].imshow(p, interpolation="nearest", cmap='RdYlBu')
        else:
            ax[0,i].imshow(p, interpolation="nearest", cmap='brg')
        ax[0,i].set_title("S{0}".format(i))
        ax[0,i].axis("off")

    for i in range(len(state_sequence)):
        overlap_list = patternLib.compute_overlap_list(state_sequence[i], pattern_list)
        ax[1, i].bar(range(len(overlap_list)), overlap_list)
        if ref_idx is None:
            max_val = np.max(overlap_list)
            max_idx = np.argmax(overlap_list)
            ax[1,i].set_title(f"m = {round(max_val,2)} \n with P{max_idx}")
        else:
            ax[1, i].set_title("m = {1}".format(i, round(overlap_list[ref_idx], 2)))
        ax[1, i].set_ylim([-1, 1])
        ax[1, i].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        if i > 0: 
            ax[1, i].set_xticklabels([])
            ax[1, i].set_yticklabels([])

    if suptitle is not None:
        f.suptitle(suptitle)

    plt.show()
