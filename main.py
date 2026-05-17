import numpy as np
import matplotlib.pyplot as plt

# 1. Исходные параметры модели
A_deg = 60
A_rad = np.radians(A_deg)

spectrum = [
    {"name": "Фиолетовый", "lambda": 400, "n": 1.532, "color": "violet"},
    {"name": "Синий", "lambda": 450, "n": 1.528, "color": "blue"},
    {"name": "Голубой", "lambda": 490, "n": 1.524, "color": "cyan"},
    {"name": "Зелёный", "lambda": 530, "n": 1.519, "color": "green"},
    {"name": "Жёлтый", "lambda": 580, "n": 1.515, "color": "yellow"},
    {"name": "Оранжевый", "lambda": 620, "n": 1.512, "color": "orange"},
    {"name": "Красный", "lambda": 700, "n": 1.508, "color": "red"},
]

# 2. Расчёт углов отклонения
for item in spectrum:
    delta_rad = (item["n"] - 1) * A_rad
    delta_deg = np.degrees(delta_rad)

    item["delta_rad"] = delta_rad
    item["delta_deg"] = delta_deg


# 3. Вывод результатов в консоль
print("Результаты моделирования")
print(f"Угол призмы A = {A_deg}°")
print()
print("Цвет\t\tДлина волны, нм\t n(λ)\t Угол отклонения, град")

for item in spectrum:
    print(
        f"{item['name']:12s}\t"
        f"{item['lambda']:>5}\t\t"
        f"{item['n']:.3f}\t"
        f"{item['delta_deg']:.2f}"
    )

delta_violet = spectrum[0]["delta_deg"]
delta_red = spectrum[-1]["delta_deg"]
delta_difference = delta_violet - delta_red

print()
print("Угловая ширина спектра:")
print(f"Δδ = {delta_violet:.2f}° - {delta_red:.2f}° = {delta_difference:.2f}°")


# 4. Построение графика
plt.figure(figsize=(12, 7))


# 5. Призма
prism_x = [0, 2.2, 1.1, 0]
prism_y = [0, 0, 2.2, 0]

plt.plot(prism_x, prism_y, color="black", linewidth=2.5)
plt.fill(prism_x, prism_y, color="lightgray", alpha=0.45)

plt.text(0.78, 0.75, "Призма", fontsize=15)


# 6. Входящий белый луч
plt.plot(
    [-2.6, 0.55],
    [1.1, 1.1],
    color="black",
    linewidth=3.5
)

plt.text(-2.55, 1.24, "Белый свет", fontsize=14)


# 7. Луч внутри призмы
inside_start = (0.55, 1.1)
inside_end = (1.55, 1.0)

plt.plot(
    [inside_start[0], inside_end[0]],
    [inside_start[1], inside_end[1]],
    color="white",
    linewidth=4
)


# 8. Выходящие цветные лучи
start_x = inside_end[0]
start_y = inside_end[1]

screen_x = 5.4

# Координаты попадания лучей на экран, специально заданы с небольшим расхождением, чтобы спектр был хорошо виден на рисунке.
screen_y_values = [0.35, 0.23, 0.11, -0.01, -0.13, -0.25, -0.37]

for item, screen_y in zip(spectrum, screen_y_values):
    item["screen_y"] = screen_y

    plt.plot(
        [start_x, screen_x],
        [start_y, screen_y],
        color=item["color"],
        linewidth=2.8,
        label=f"{item['name']}: λ={item['lambda']} нм, δ={item['delta_deg']:.2f}°"
    )


# 9. Экран и спектр на экране
screen_y_min = min(screen_y_values) - 0.2
screen_y_max = max(screen_y_values) + 0.2

plt.plot(
    [screen_x, screen_x],
    [screen_y_min, screen_y_max],
    color="black",
    linewidth=2.5
)

plt.text(screen_x - 0.25, screen_y_max + 0.12, "Экран", fontsize=14)

for item in spectrum:
    y = item["screen_y"]

    plt.plot(
        [screen_x - 0.08, screen_x + 0.08],
        [y, y],
        color=item["color"],
        linewidth=8
    )


# 10. Оформление
plt.title("Разложение белого света в призме", fontsize=17)
plt.xlabel("x", fontsize=13)
plt.ylabel("y", fontsize=13)

plt.grid(True, alpha=0.3)
plt.axis("equal")

plt.xlim(-2.9, 5.9)
plt.ylim(-1.2, 2.6)

plt.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("prism_spectrum.png", dpi=300, bbox_inches="tight")
plt.show()