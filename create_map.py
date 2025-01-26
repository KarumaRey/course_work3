import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from skimage import measure
from shapely.geometry import LineString

top = 1.0
bottom = -1.0


def fade(t):
    return t ** 3 * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    return a + t * (b - a)


def grad(hash, x, y):
    h = hash & 3
    return (x if h < 2 else -x) + (y if h & 1 == 0 else -y)


def perlin(x, y, permutation):
    x0 = int(np.floor(x)) & 255
    y0 = int(np.floor(y)) & 255
    x1 = (x0 + 1) & 255
    y1 = (y0 + 1) & 255

    n00 = grad(permutation[x0 + permutation[y0]], x - np.floor(x), y - np.floor(y))
    n01 = grad(permutation[x0 + permutation[y1]], x - np.floor(x), y - np.floor(y) - 1)
    n10 = grad(permutation[x1 + permutation[y0]], x - np.floor(x) - 1, y - np.floor(y))
    n11 = grad(permutation[x1 + permutation[y1]], x - np.floor(x) - 1, y - np.floor(y) - 1)

    ix0 = lerp(n00, n01, fade(y - np.floor(y)))
    ix1 = lerp(n10, n11, fade(y - np.floor(y)))

    return lerp(ix0, ix1, fade(x - np.floor(x)))


def generate_perlin_noise(width, height, scale, permutation):
    noise = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            noise[i][j] = perlin(j / scale, i / scale, permutation)
    return noise


def create_permutation():
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    return np.stack([p, p]).flatten()


def generate_contours(X, Y, Z, filename, contour_levels):
    all_data = []
    for lvl in contour_levels:
        c = measure.find_contours(Z, lvl)
        for cnt in c:
            line_coords = [(X[int(pt[0]), int(pt[1])], Y[int(pt[0]), int(pt[1])]) for pt in cnt]
            all_data.append({"geometry": LineString(line_coords), "level": lvl})

    gdf = gpd.GeoDataFrame(all_data, crs="EPSG:3857")
    gdf.to_file(filename, driver="GeoJSON")


width = 1000
height = 1000
scale = 300

permutation = create_permutation()
Z = generate_perlin_noise(width, height, scale, permutation)


x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
X, Y = np.meshgrid(x, y)
z_levels = np.linspace(bottom, top, 11) * 1000


generate_contours(X, Y, Z * 1000, "relief_contours.geojson", z_levels)


fig, ax = plt.subplots()
ax.set_facecolor('#EEEEEE')
contour = plt.contour(X, Y, Z * 1000, levels=z_levels, cmap='terrain')
plt.colorbar(contour)
plt.title('Contour Map')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()