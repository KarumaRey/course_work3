import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math


def load_contours(filename):
    gdf = gpd.read_file(filename)
    return gdf


def line_intersection(p1, p2, line):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3, x4, y4 = line

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if det == 0:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
    u = ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / det

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (intersect_x, intersect_y)

    return None


def find_intersections(p1, p2, gdf):
    intersections = []
    for _, row in gdf.iterrows():
        line = row['geometry']
        z = row['level']

        x_coords = line.xy[0]
        y_coords = line.xy[1]

        for i in range(len(x_coords) - 1):
            line_segment = (x_coords[i], y_coords[i], x_coords[i + 1], y_coords[i + 1])
            intersection_point = line_intersection(p1, p2, line_segment)

            if intersection_point:
                intersections.append((intersection_point[0], intersection_point[1], z))

    return intersections


def linear_interpolation(x, y, xi):
    for i in range(len(x) - 1):
        if x[i] <= xi <= x[i + 1]:
            return y[i] + (y[i + 1] - y[i]) * (xi - x[i]) / (x[i + 1] - x[i])
    return None


def evaluate_cubic_spline(xi, x, a, b, c, d):
    for i in range(len(x) - 1):
        if x[i] <= xi <= x[i + 1]:
            return a[i] + b[i] * (xi - x[i]) + c[i] * (xi - x[i]) ** 2 + d[i] * (xi - x[i]) ** 3
    return None


def cubic_spline_interpolation(x, y):
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]
    alpha = [(3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]) for i in range(1, n)]

    L = [1] + [2 * (h[i - 1] + h[i]) for i in range(1, n)] + [1]
    mu = [0] + [h[i] / L[i + 1] for i in range(1, n)] + [0]

    c = [0] * (n + 1)
    for i in range(1, n):
        c[i] = (alpha[i - 1] - mu[i - 1] * c[i - 1]) / L[i]

    a = y[:-1]
    b = [0] * n
    d = [0] * n

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c, d


def calculate_distances_and_angles(distances_sorted, heights_sorted, plt):
    for i in range(1, len(distances_sorted)):
        x1, z1 = distances_sorted[i - 1], heights_sorted[i - 1]
        x2, z2 = distances_sorted[i], heights_sorted[i]

        distance = (x2 - x1)
        angle = math.degrees(math.atan2(z2 - z1, x2 - x1))

        if distance < 10:
            continue
        mid_x = (x1 + x2) / 2
        mid_y = max(z1, z2) / 2 - 500
        plt.text(mid_x, mid_y, f'd={distance:.1f} м\nθ={angle:.2f}°', fontsize=8, ha='center')


def plot_intersection_graph(distances_sorted, heights_sorted, p1, p2):
    heights_sorted = list(heights_sorted)
    distances_sorted = list(distances_sorted)

    if heights_sorted:
        height_p1 = heights_sorted[0]
        height_p2 = heights_sorted[-1]
    else:
        height_p1 = linear_interpolation(distances_sorted, heights_sorted, 0) if distances_sorted else 0
        height_p2 = linear_interpolation(distances_sorted, heights_sorted,
                                         np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

    dist_full = [0] + distances_sorted + [np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)]
    heights_full = [height_p1] + heights_sorted + [height_p2]

    control_points = np.array([dist_full, heights_full]).T
    bezier_points = compute_bezier(control_points)

    plt.figure()
    plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'r-', label='Кривая Безье')

    for i in range(len(heights_full)):
        plt.plot([dist_full[i], dist_full[i]], [heights_full[i], heights_full[i]], 'ko--', linewidth=2)

    plt.title('График аппроксимации высоты точек пересечения')
    plt.xlabel('Расстояние')
    plt.ylabel('Высота (z)')
    plt.ylim(-1100, 1100)
    plt.grid()
    plt.legend()
    plt.show()

    distances_sorted, heights_sorted = dist_full, heights_full
    plt.figure()
    calculate_distances_and_angles(distances_sorted, heights_sorted, plt)
    plt.plot(distances_sorted, heights_sorted, 'bo-', label='Пересечения')
    for i in range(len(heights_sorted)):
        plt.plot([distances_sorted[i], distances_sorted[i]], [-1100, heights_sorted[i]], 'k--', linewidth=2)
    plt.title('График высоты точек пересечения')
    plt.xlabel('Расстояние')
    plt.ylabel('Высота (z)')
    plt.ylim(-1100, 1100)
    plt.grid()
    plt.legend()
    plt.show()

    a, b, c, d = cubic_spline_interpolation(distances_sorted, heights_sorted)

    x_interp = np.linspace(0, distances_sorted[-1], 100)
    y_interp = [evaluate_cubic_spline(xi, distances_sorted, a, b, c, d) for xi in x_interp]

    plt.figure()
    plt.plot(x_interp, y_interp, 'g-', label='Интерполяция методом кубических сплайнов')
    plt.plot(distances_sorted, heights_sorted, 'bo', label='Точки пересечения')
    calculate_distances_and_angles(distances_sorted, heights_sorted, plt)

    for i in range(len(heights_sorted)):
        plt.plot([distances_sorted[i], distances_sorted[i]], [-1100, heights_sorted[i]], 'k--', linewidth=2)

    plt.title('Интерполяция высоты точек пересечения')
    plt.xlabel('Расстояние')
    plt.ylabel('Высота (z)')
    plt.ylim(-1100, 1100)
    plt.grid()
    plt.legend()
    plt.show()


def compute_bezier(control_points):
    n = len(control_points) - 1
    num_points = 100
    t = np.linspace(0, 1, num_points)
    bezier_curve = np.zeros((num_points, 2))

    for i in range(n + 1):
        binomial_coeff = math.comb(n, i)
        bezier_curve += np.outer(
            binomial_coeff * (t ** i) * ((1 - t) ** (n - i)),
            control_points[i]
        )
    return bezier_curve


points = []
gdf = load_contours("relief_contours.geojson")


def onclick(event):
    global points
    if event.xdata is not None and event.ydata is not None:
        points.append((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        plt.draw()

        if len(points) == 2:
            p1, p2 = points

            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--')
            plt.legend()

            intersections = find_intersections(p1, p2, gdf)

            i = 0
            for x, y, z in intersections:
                plt.plot(x, y, 'ro', markersize=5)

            distances = []
            heights = []
            for x, y, z in intersections:
                distance = np.sqrt((x - p1[0]) ** 2 + (y - p1[1]) ** 2)
                distances.append(distance)
                heights.append(z)
                print(distance, "   ", z)

            if distances:
                sorted_data = sorted(zip(distances, heights))
                distances_sorted, heights_sorted = zip(*sorted_data)

                plot_intersection_graph(distances_sorted, heights_sorted, p1, p2)

            points = []


fig, ax = plt.subplots()
ax.set_facecolor('#EEEEEE')
plt.title('Кликните для выбора концов отрезка')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()

for _, row in gdf.iterrows():
    line = row['geometry']
    x, y = line.xy
    plt.plot(x, y, color='black')

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()