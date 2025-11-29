from math import acos, degrees
A = 80
B = 80

x_t = 100
y_t = -60
z_t = 80

y_fin = y_t + 23

AB = (x_t ** 2 + z_t ** 2) ** 0.5
k = 40 / AB

EF = k * x_t
BF = k * z_t

z_fin = z_t - BF
x_fin = x_t - EF


R = (x_fin ** 2 + z_fin ** 2) ** 0.5 # угол поворота основания
L = (x_fin ** 2 + y_fin ** 2 + z_fin ** 2) ** 0.5 # расстояние от начала манипулятора до точки
beta = acos((A ** 2 + B ** 2 - L ** 2) / (2 * A * B))

epsilon = acos((A ** 2 + L ** 2 - B ** 2) / (2 * L * A))
delta = acos(R / L)

alpha = epsilon - delta

gamma = degrees(acos(x_fin / R))

if gamma <= 90:
    print(f'Импульс основания равен: {560 + (gamma) * 10.7777}')
elif gamma > 90:
    print(f'Импульс основания равен: {1530 + 11.22 * (gamma - 90)}')

print(f'Импульс предплечья равен: {326 + 11.09 * (180 - degrees(alpha) - degrees(beta))}')

print('\nИтог\n')

#print(f'Импульс основания равен: {500 + (gamma + 7) * 10.5555555 - 6}')
print(f'Импульс плеча равен: {500 + ((180 - degrees(alpha) + 5) * 10.5555555)}')
#print(f'Импульс предплечья равен: {500 + ((180 - degrees(beta) - degrees(alpha) + 24) * 10.555555)}')
#print()
#print(f'Угол основания равен: {gamma + 7}')
#print(f'Угол плеча равен: {(180 - degrees(alpha) + 5)}')
#print(f'Угол предплечья равен: {180 - degrees(beta) - degrees(alpha) + 24}')
#print(x_fin)
#print(y_fin)
#print(z_fin)
