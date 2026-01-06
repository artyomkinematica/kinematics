from math import cos, sin, radians


class ManipulatorKinematics:
    def __init__(self):
        # Длины звеньев (мм)
        self.A = 80  # Плечо
        self.B = 80  # Предплечье

        # Смещения
        self.base_height = 60  # Высота основания
        self.tool_offset = 23  # Смещение инструмента

        # Коэффициенты для преобразования углов в импульсы
        self.pulse_per_degree = 10.5555555
        self.center_pulse = 500

    def inverse_kinematics(self, x_t, y_t, z_t):
        """Обратная кинематика (из вашего кода)"""
        y_fin = y_t + self.tool_offset

        AB = (x_t ** 2 + z_t ** 2) ** 0.5
        k = 40 / AB

        EF = k * x_t
        BF = k * z_t

        z_fin = z_t - BF
        x_fin = x_t - EF

        R = (x_fin ** 2 + z_fin ** 2) ** 0.5
        L = (x_fin ** 2 + y_fin ** 2 + z_fin ** 2) ** 0.5

        from math import acos, degrees
        beta = acos((self.A ** 2 + self.B ** 2 - L ** 2) / (2 * self.A * self.B))
        epsilon = acos((self.A ** 2 + L ** 2 - self.B ** 2) / (2 * L * self.A))
        delta = acos(R / L)

        alpha = epsilon - delta
        gamma = degrees(acos(x_fin / R))

        # Углы суставов
        base_angle = gamma + 11
        shoulder_angle = 180 - degrees(alpha)
        elbow_angle = 180 - degrees(beta) - degrees(alpha)

        # Импульсы сервоприводов
        base_pulse = self.center_pulse + base_angle * self.pulse_per_degree
        shoulder_pulse = self.center_pulse + shoulder_angle * self.pulse_per_degree
        elbow_pulse = self.center_pulse + elbow_angle * self.pulse_per_degree

        return {
            'angles': {
                'base': base_angle,
                'shoulder': shoulder_angle,
                'elbow': elbow_angle
            },
            'pulses': {
                'base': base_pulse,
                'shoulder': shoulder_pulse,
                'elbow': elbow_pulse
            }
        }

    def forward_kinematics(self, base_angle, shoulder_angle, elbow_angle):
        """Прямая кинематика - вычисление координат по углам суставов"""

        # Преобразуем углы в радианы
        base_rad = radians(base_angle - 11)  # Учитываем смещение основания
        shoulder_rad = radians(shoulder_angle)
        elbow_rad = radians(elbow_angle)

        # Вычисляем координаты плечевого сустава
        shoulder_x = 0  # Плечо начинается в начале координат по X
        shoulder_y = self.base_height  # Высота основания
        shoulder_z = 0  # Плечо начинается в начале координат по Z

        # Координаты локтевого сустава
        elbow_x = self.A * cos(shoulder_rad) * cos(base_rad)
        elbow_y = self.base_height + self.A * sin(shoulder_rad)
        elbow_z = self.A * cos(shoulder_rad) * sin(base_rad)

        # Суммарный угол для предплечья
        forearm_angle = shoulder_angle + elbow_angle - 180

        # Координаты конца манипулятора (до учета инструмента)
        end_x = elbow_x + self.B * cos(radians(forearm_angle)) * cos(base_rad)
        end_y = elbow_y + self.B * sin(radians(forearm_angle))
        end_z = elbow_z + self.B * cos(radians(forearm_angle)) * sin(base_rad)

        # Корректируем координаты с учетом смещения инструмента
        tool_y = end_y - self.tool_offset

        return {
            'shoulder': (shoulder_x, shoulder_y, shoulder_z),
            'elbow': (elbow_x, elbow_y, elbow_z),
            'end_effector': (end_x, tool_y, end_z),
            'tool_point': (end_x, end_y, end_z)
        }

    def calculate_similar_triangles_point(self, target_x, target_z, offset=40):
        """
        Метод определения точной координаты через подобные треугольники
        Используется для компенсации смещения инструмента
        """
        # Вычисляем расстояние до цели в плоскости XZ
        distance = (target_x ** 2 + target_z ** 2) ** 0.5

        if distance == 0:
            return target_x, target_z

        # Коэффициент подобия треугольников
        k = offset / distance

        # Корректируем координаты
        corrected_x = target_x - k * target_x
        corrected_z = target_z - k * target_z

        return corrected_x, corrected_z

    def full_kinematic_cycle(self, x, y, z):
        """Полный цикл: обратная + прямая кинематика для проверки"""
        print(f"Целевая точка: ({x}, {y}, {z})")

        # Обратная кинематика
        ik_result = self.inverse_kinematics(x, y, z)
        print(f"\nОбратная кинематика:")
        print(f"Углы: основание={ik_result['angles']['base']:.2f}°, "
              f"плечо={ik_result['angles']['shoulder']:.2f}°, "
              f"локоть={ik_result['angles']['elbow']:.2f}°")

        # Прямая кинематика
        fk_result = self.forward_kinematics(
            ik_result['angles']['base'],
            ik_result['angles']['shoulder'],
            ik_result['angles']['elbow']
        )

        print(f"\nПрямая кинематика:")
        print(f"Плечо: {fk_result['shoulder']}")
        print(f"Локоть: {fk_result['elbow']}")
        print(f"Конец манипулятора: {fk_result['end_effector']}")

        # Проверка точности
        error_x = abs(fk_result['end_effector'][0] - x)
        error_y = abs(fk_result['end_effector'][1] - y)
        error_z = abs(fk_result['end_effector'][2] - z)
        total_error = (error_x ** 2 + error_y ** 2 + error_z ** 2) ** 0.5

        print(f"\nТочность:")
        print(f"Ошибка: X={error_x:.2f}, Y={error_y:.2f}, Z={error_z:.2f}")
        print(f"Общая ошибка: {total_error:.2f} мм")

        return ik_result, fk_result


# Пример использования
if __name__ == "__main__":
    manipulator = ManipulatorKinematics()

    # Тестовая точка из вашего кода
    x, y, z = 100, -60, 80

    # Полный цикл кинематики
    ik, fk = manipulator.full_kinematic_cycle(x, y, z)

    # Дополнительно: расчет точки через подобные треугольники
    print(f"\nМетод подобных треугольников:")
    corrected_x, corrected_z = manipulator.calculate_similar_triangles_point(x, z)
    print(f"Исходные координаты: X={x}, Z={z}")
    print(f"Скорректированные: X={corrected_x:.2f}, Z={corrected_z:.2f}")

    # Прямая кинематика для произвольных углов
    print(f"\nПрямая кинематика для углов (90°, 90°, 90°):")
    test_fk = manipulator.forward_kinematics(90, 90, 90)
    print(f"Конечная точка: {test_fk['end_effector']}")