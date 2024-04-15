import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def get_random_sign():
    return np.random.choice([-1, 1])

def CosineAngle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    cosine = dot_product / (magnitude1 * magnitude2)
    return cosine

def GenerateRandomPolygon(pointOnCircle, seed, eccentricity):
    np.random.seed(seed)
    vertices_random = []
    vertices = []
    r = 200
    for _ in range(pointOnCircle):
        sin = np.random.uniform(-1, 1)
        cos = np.sqrt(1 - sin ** 2)
        vertices_random.append(Point(get_random_sign() * r * sin, get_random_sign() * r * cos))
    pointMinX = Point(500,0)
    for point in vertices_random:
        if point.x < pointMinX.x:
            pointMinX = point
    vertices.append(pointMinX)
    p1 = Point(pointMinX.x, pointMinX.y - 50)
    p2 = pointMinX
    condition = True
    while condition:
        minCos = 1
        for point in vertices_random:
            if not (p1.x == point.x and p1.y == point.y) and not (p2.x == point.x and p2.y == point.y):
                if CosineAngle(p1, p2, point) < minCos:
                    minCos = CosineAngle(p1, p2, point)
                    pointWithMinCos = point
        vertices.append(pointWithMinCos)
        p1 = p2
        p2 = pointWithMinCos
        condition = not (p2.x == pointMinX.x and p2.y == pointMinX.y)
    return vertices

def calculate_triangle_area(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p1.x - p3.x, p1.y - p3.y])
    area = abs(np.cross(v1, v2)) / 2
    return area

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Кількість точок на колі
    pointOnCircle = 20

    # Насіння для випадкового генератора
    seed = 45

    # Генеруємо опуклий многокутник
    vertices = GenerateRandomPolygon(pointOnCircle, seed, 0)

    # Визначення кількості точок на процес
    points_per_process = len(vertices) // size
    remainder = len(vertices) % size

    # Підрахунок точок на процес, включаючи залишок
    points_counts = [points_per_process + 1 if i < remainder else points_per_process for i in range(size)]

    # Визначення початкового та кінцевого індексу для кожного процесу
    start_index = sum(points_counts[:rank])
    end_index = start_index + points_counts[rank]

    # Локальний обчислення площі для кожного процесу
    local_total_area = 0
    for i in range(start_index + 1, end_index - 1):
        local_total_area += calculate_triangle_area(vertices[0], vertices[i], vertices[i + 1])

    # Збір локальних результатів на кореневому процесі
    total_areas = comm.gather(local_total_area, root=0)

    # Обчислення загальної площі на кореневому процесі
    if rank == 0:
        total_area = sum(total_areas)
        print("Загальна площа трикутників:", total_area)

        # Візуалізація опуклої оболонки
        x = [point.x for point in vertices]
        y = [point.y for point in vertices]
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, 'bo-')

        # Показ візуалізації
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Convex Hull Visualization with MPI')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
