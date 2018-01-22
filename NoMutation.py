import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation

def breed(genotypes):
    new_genotype = genotypes.copy()
    for i in range(len(genotypes)):
        new_genotype[i] = np.random.choice(genotypes, 1)
    return new_genotype

def test_fixation(population_size):
    genotypes = np.zeros(population_size)
    genotypes[np.random.randint(0, len(genotypes))] = 1
    while True:
        new_genotype = breed(genotypes)
        if np.all(np.equal(new_genotype, genotypes)):
            break
        genotypes = new_genotype
    if 1 in genotypes:
        return 1
    else:
        return 0

def test_all_fixation():
    res = []
    for population_size in range(10, 100, 5):
        print(population_size)
        percentage_fixed = []
        for _ in range(1000):
            percentage_fixed.append(test_fixation(population_size))
        res.append(sum(percentage_fixed)/len(percentage_fixed))
    plt.plot([i for i in range(10, 100, 5)], res)
    plt.show()

def iterate_histogram(time_steps):
    global histogram_data
    histogram_data = []
    population_size = 30
    n_populations = 1000
    genotypes = np.zeros((population_size, n_populations))
    for i in range(n_populations):
        genotypes[np.random.randint(0, genotypes.shape[0]), i] = 1
    for h in range(time_steps):
        print(h)
        for i, row in enumerate(genotypes.transpose()):
            genotypes[:, i] = breed(row)
        n, bins = np.histogram(np.sum(genotypes, axis=0), range=(0, population_size), bins=[i-0.5 for i in range(population_size+2)])
        histogram_data.append((n, bins))


def make_animated_histogram():
    # histogram our data with numpy
    global histogram_data
    time_steps = 100
    iterate_histogram(time_steps)
    n = histogram_data[0][0]
    bins = histogram_data[0][1]

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    nrects = len(left)

    nverts = nrects * (1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom

    patch = None

    def animate(i):
        # simulate new data coming in
        data = np.random.randn(1000)
        n, bins = np.histogram(data, 100)
        n = histogram_data[i][0]
        top = bottom + n
        verts[1::5, 1] = top
        verts[2::5, 1] = top
        return [patch, ]

    fig, ax = plt.subplots()
    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(
        barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())
    print(histogram_data)
    ani = animation.FuncAnimation(fig, animate, time_steps, repeat=False, blit=True, interval=200)
    ani.save('FixationNoMutation.mp4')

make_animated_histogram()