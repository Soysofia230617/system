#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <limits>

const double G = 6.67430e-11;          // Гравитационная постоянная, м^3 кг^-1 с^-2
const double a0 = 1.2e-10;             // MOND константа, м/с^2
const double DT = 1e9;                 // Уменьшенный шаг времени, с (~0.00003 млн лет)
const int N_STEPS = 20000;             // Количество шагов
const double R_d = 3e19;               // Масштабный радиус диска, м (~3 кпк)
const double R_bulge = 5e18;           // Масштабный радиус балджа, м (~0.5 кпк)
const double M_TOTAL = 2e40;           // Общая масса галактики, кг (~10^10 M_sun)
const double M_BULGE_FRACTION = 0.2;   // Доля массы в балдже
const double EPSILON = 0.05;           // Амплитуда спирального возмущения
const double ALPHA = 15 * M_PI / 180;  // Угол наклона спирали, радианы
const int M_ARMS = 2;                  // Число спиральных рукавов
const double R_0 = 1e19;               // Опорный радиус спирали, м (~1 кпк)
const double YEAR = 3.15576e7;         // Секунд в году
const double KPC = 3.086e19;           // Метров в килопарсеке
const double MAX_ACC = 2e-7;           // Ограничение ускорения, м/с^2
const int RELAX_STEPS = 150;           // Количество шагов релаксации
const double DAMPING_FACTOR = 0.97;    // Менее агрессивное демпфирование
const double SIGMA_V_FACTOR = 0.015;   // Уменьшенный фактор дисперсии скоростей
const double SIGMA_0 = 3.44e-18;       // Центральная поверхностная плотность, кг/м^2
const double SOFTENING = 3e19;         // Смягчение, м (~1 R_d)

// Структура для звезды
struct Star {
    double x, y;      // Позиция, м
    double vx, vy;    // Скорость, м/с
    double mass;      // Масса, кг
};

// Функция MOND-интерполяции
double mond_interpolation(double x) {
    return x / (1.0 + x);
}

// Функция MOND-ускорения
double mond_acceleration(double a_newton) {
    if (std::isnan(a_newton) || std::isinf(a_newton)) return 0.0;
    double x = std::abs(a_newton) / a0;
    if (x < 1e-6) return a_newton;
    double mu = mond_interpolation(x);
    if (mu < 1e-6) return a_newton;
    return a_newton / mu;
}

// Функция MOND-потенциала
double mond_potential(double r, double m) {
    double r_soft = sqrt(r * r + SOFTENING * SOFTENING);
    double a_newton = G * m / (r_soft * r_soft);
    double a_mond = mond_acceleration(a_newton);
    // Численное интегрирование: phi = -∫ a_mond dr
    double phi = 0.0;
    const double dr = r_soft / 100.0;
    for (double ri = r_soft; ri < 1e21; ri += dr) {
        double a_n = G * m / (ri * ri);
        double a_m = mond_acceleration(a_n);
        phi -= a_m * dr;
    }
    if (std::isnan(phi) || std::isinf(phi)) {
        phi = -G * m / r_soft;
    }
    return phi;
}

// Вычисление массы внутри радиуса
double enclosed_mass(double r, bool is_bulge) {
    if (is_bulge) {
        double sigma = R_bulge;
        return M_TOTAL * M_BULGE_FRACTION * (1.0 - exp(-r * r / (2.0 * sigma * sigma)));
    } else {
        return M_TOTAL * (1.0 - M_BULGE_FRACTION) * (1.0 - exp(-r / R_d) * (1.0 + r / R_d));
    }
}

// Вычисление ускорений
void compute_accelerations(const std::vector<Star>& stars, std::vector<double>& ax, std::vector<double>& ay) {
    int n = stars.size();
    ax.assign(n, 0.0);
    ay.assign(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double dx = stars[j].x - stars[i].x;
            double dy = stars[j].y - stars[i].y;
            double r = sqrt(dx * dx + dy * dy);
            double r_soft = sqrt(r * r + SOFTENING * SOFTENING);

            double a_newton = G * stars[j].mass / (r_soft * r_soft);
            double a_mond = mond_acceleration(a_newton);
            if (a_mond > MAX_ACC || std::isnan(a_mond) || std::isinf(a_mond)) a_mond = MAX_ACC;

            ax[i] += a_mond * dx / r_soft;
            ay[i] += a_mond * dy / r_soft;
        }
    }
}

// Генерация начальных условий
std::vector<Star> generate_galaxy(int n_stars, unsigned seed) {
    std::vector<Star> stars(n_stars);
    std::mt19937 gen(seed);
    std::exponential_distribution<double> r_disk_dist(1.0 / R_d);
    std::normal_distribution<double> r_bulge_dist(0.0, R_bulge);
    std::uniform_real_distribution<double> phi_dist(0, 2 * M_PI);
    int n_bulge = static_cast<int>(n_stars * M_BULGE_FRACTION);
    int n_disk = n_stars - n_bulge;

    for (int i = 0; i < n_stars; ++i) {
        double r, phi;
        bool is_bulge = (i < n_bulge);
        if (is_bulge) {
            r = std::abs(r_bulge_dist(gen));
        } else {
            do {
                r = r_disk_dist(gen);
                phi = phi_dist(gen);
                double spiral_term = cos(M_ARMS * (phi - log(r / R_0) / tan(ALPHA)));
                double prob = 1.0 + EPSILON * spiral_term;
                if (prob > gen() / gen.max()) break;
            } while (true);
        }
        phi = phi_dist(gen);

        stars[i].x = r * cos(phi);
        stars[i].y = r * sin(phi);

        // Вириальная калибровка скорости
        double m_enc = enclosed_mass(r, is_bulge);
        double a_newton = G * m_enc / (r * r + SOFTENING * SOFTENING);
        double a_mond = mond_acceleration(a_newton);
        double v_circ = sqrt(a_mond * r);
        if (std::isnan(v_circ) || std::isinf(v_circ)) v_circ = 150000.0;

        // Дисперсия скоростей
        std::normal_distribution<double> v_disp(0.0, SIGMA_V_FACTOR * v_circ);
        double v_r = v_disp(gen);
        double v_t = v_circ + v_disp(gen);

        stars[i].vx = -v_t * sin(phi) + v_r * cos(phi);
        stars[i].vy = v_t * cos(phi) + v_r * sin(phi);

        // Масса пропорциональна поверхностной плотности
        double sigma = is_bulge ? SIGMA_0 : SIGMA_0 * exp(-r / R_d);
        stars[i].mass = M_TOTAL / n_stars;
    }
    return stars;
}

// Корректировка центра масс
void correct_center_of_mass(std::vector<Star>& stars) {
    double cm_x = 0.0, cm_y = 0.0;
    double cm_vx = 0.0, cm_vy = 0.0;
    double total_mass = 0.0;

    for (const auto& star : stars) {
        cm_x += star.mass * star.x;
        cm_y += star.mass * star.y;
        cm_vx += star.mass * star.vx;
        cm_vy += star.mass * star.vy;
        total_mass += star.mass;
    }

    cm_x /= total_mass;
    cm_y /= total_mass;
    cm_vx /= total_mass;
    cm_vy /= total_mass;

    for (auto& star : stars) {
        star.x -= cm_x;
        star.y -= cm_y;
        star.vx -= cm_vx;
        star.vy -= cm_vy;
    }
}

// Корректировка момента импульса
void correct_angular_momentum(std::vector<Star>& stars, double L_z_target) {
    double L_z_current = 0.0;
    double total_mass = 0.0;
    for (const auto& star : stars) {
        L_z_current += star.mass * (star.x * star.vy - star.y * star.vx);
        total_mass += star.mass;
    }

    if (L_z_current == 0.0) return;

    double factor = L_z_target / L_z_current;
    for (auto& star : stars) {
        star.vx *= factor;
        star.vy *= factor;
    }
}

// Обновление позиций и скоростей
void update_stars(std::vector<Star>& stars, std::vector<double>& ax, std::vector<double>& ay, double L_z_initial) {
    int n = stars.size();
    std::vector<double> ax_new(n), ay_new(n);

    for (int i = 0; i < n; ++i) {
        stars[i].vx += 0.5 * DT * ax[i];
        stars[i].vy += 0.5 * DT * ay[i];
    }

    for (int i = 0; i < n; ++i) {
        stars[i].x += DT * stars[i].vx;
        stars[i].y += DT * stars[i].vy;
    }

    compute_accelerations(stars, ax_new, ay_new);

    for (int i = 0; i < n; ++i) {
        stars[i].vx += 0.5 * DT * ax_new[i];
        stars[i].vy += 0.5 * DT * ay_new[i];
        ax[i] = ax_new[i];
        ay[i] = ay_new[i];
    }
}

// Релаксация начальных условий
void relax_initial_conditions(std::vector<Star>& stars, std::vector<double>& ax, std::vector<double>& ay, double L_z_initial) {
    std::ofstream relax_out("relaxation_log.txt");
    relax_out << "Step E_kinetic_J E_potential_J E_total_J V_disp_m_s L_z_kg_m2_s Phi_avg_J_kg\n";

    double v_mean_x, v_mean_y, v_disp_x, v_disp_y;

    for (int step = 0; step < RELAX_STEPS; ++step) {
        update_stars(stars, ax, ay, L_z_initial);

        v_mean_x = 0.0;
        v_mean_y = 0.0;
        int n = stars.size();
        for (const auto& star : stars) {
            v_mean_x += star.vx / n;
            v_mean_y += star.vy / n;
        }
        for (auto& star : stars) {
            star.vx = DAMPING_FACTOR * (star.vx - v_mean_x) + v_mean_x;
            star.vy = DAMPING_FACTOR * (star.vy - v_mean_y) + v_mean_y;
        }

        // Корректировка L_z каждые 10 шагов
        if (step % 10 == 0) {
            correct_angular_momentum(stars, L_z_initial);
        }

        double E_k = 0.0, E_p = 0.0, L_z = 0.0, phi_avg = 0.0;
        v_mean_x = 0.0;
        v_mean_y = 0.0;
        v_disp_x = 0.0;
        v_disp_y = 0.0;
        int phi_count = 0;
        for (const auto& star : stars) {
            double v2 = star.vx * star.vx + star.vy * star.vy;
            E_k += 0.5 * star.mass * v2;
            L_z += star.mass * (star.x * star.vy - star.y * star.vx);
            v_mean_x += star.vx / n;
            v_mean_y += star.vy / n;
        }
        for (const auto& star : stars) {
            v_disp_x += (star.vx - v_mean_x) * (star.vx - v_mean_x) / n;
            v_disp_y += (star.vy - v_mean_y) * (star.vy - v_mean_y) / n;
        }
        double v_disp = sqrt(v_disp_x + v_disp_y);

        for (size_t i = 0; i < stars.size(); ++i) {
            for (size_t j = i + 1; j < stars.size(); ++j) {
                double dx = stars[j].x - stars[i].x;
                double dy = stars[j].y - stars[i].y;
                double r = sqrt(dx * dx + dy * dy);
                double phi = mond_potential(r, stars[j].mass);
                if (!std::isnan(phi) && !std::isinf(phi)) {
                    E_p += stars[i].mass * phi;
                    phi_avg += phi;
                    phi_count++;
                }
            }
        }
        E_p *= 2.0; // Учет симметрии
        phi_avg = phi_count > 0 ? phi_avg / phi_count : 0.0;

        double E_total = E_k + E_p;
        relax_out << step << " " << E_k << " " << E_p << " " << E_total << " " << v_disp << " " << L_z << " " << phi_avg << "\n";

        if (step % 10 == 0) {
            std::cout << "Relaxation step: " << step
                      << ", E_total: " << E_total << " J"
                      << ", E_kinetic: " << E_k << " J"
                      << ", E_potential: " << E_p << " J"
                      << ", V_disp: " << v_disp << " m/s"
                      << ", L_z: " << L_z << " kg·m²/s\n";
        }
    }

    // Финальная корректировка L_z
    correct_angular_momentum(stars, L_z_initial);

    double E_k = 0.0, E_p = 0.0, L_z = 0.0, phi_avg = 0.0;
    v_mean_x = 0.0;
    v_mean_y = 0.0;
    v_disp_x = 0.0;
    v_disp_y = 0.0;
    int phi_count = 0;
    int n = stars.size();
    for (const auto& star : stars) {
        double v2 = star.vx * star.vx + star.vy * star.vy;
        E_k += 0.5 * star.mass * v2;
        L_z += star.mass * (star.x * star.vy - star.y * star.vx);
        v_mean_x += star.vx / n;
        v_mean_y += star.vy / n;
    }
    for (const auto& star : stars) {
        v_disp_x += (star.vx - v_mean_x) * (star.vx - v_mean_x) / n;
        v_disp_y += (star.vy - v_mean_y) * (star.vy - v_mean_y) / n;
    }
    double v_disp = sqrt(v_disp_x + v_disp_y);

    for (size_t i = 0; i < stars.size(); ++i) {
        for (size_t j = i + 1; j < stars.size(); ++j) {
            double dx = stars[j].x - stars[i].x;
            double dy = stars[j].y - stars[i].y;
            double r = sqrt(dx * dx + dy * dy);
            double phi = mond_potential(r, stars[j].mass);
            if (!std::isnan(phi) && !std::isinf(phi)) {
                E_p += stars[i].mass * phi;
                phi_avg += phi;
                phi_count++;
            }
        }
    }
    E_p *= 2.0;
    phi_avg = phi_count > 0 ? phi_avg / phi_count : 0.0;

    double E_total = E_k + E_p;
    relax_out << RELAX_STEPS << " " << E_k << " " << E_p << " " << E_total << " " << v_disp << " " << L_z << " " << phi_avg << "\n";

    relax_out.close();
}

// Вычисление физических величин
void compute_physical_quantities(const std::vector<Star>& stars, int step, std::ofstream& out_quantities, double& E_total_0, double& L_z_0, double& cm_x_0, double& cm_y_0, double& cm_vx_0, double& cm_vy_0) {
    double E_k = 0.0, E_p = 0.0, L_z = 0.0;
    double v_mean_x = 0.0, v_mean_y = 0.0;
    double v_disp_x = 0.0, v_disp_y = 0.0;
    double cm_x = 0.0, cm_y = 0.0;
    double cm_vx = 0.0, cm_vy = 0.0;
    double total_mass = 0.0;
    int n = stars.size();

    for (const auto& star : stars) {
        double v2 = star.vx * star.vx + star.vy * star.vy;
        E_k += 0.5 * star.mass * v2;
        L_z += star.mass * (star.x * star.vy - star.y * star.vx);
        v_mean_x += star.vx / n;
        v_mean_y += star.vy / n;
        cm_x += star.mass * star.x;
        cm_y += star.mass * star.y;
        cm_vx += star.mass * star.vx;
        cm_vy += star.mass * star.vy;
        total_mass += star.mass;
    }

    cm_x /= total_mass;
    cm_y /= total_mass;
    cm_vx /= total_mass;
    cm_vy /= total_mass;

    for (const auto& star : stars) {
        v_disp_x += (star.vx - v_mean_x) * (star.vx - v_mean_x) / n;
        v_disp_y += (star.vy - v_mean_y) * (star.vy - v_mean_y) / n;
    }
    double v_disp = sqrt(v_disp_x + v_disp_y);

    for (size_t i = 0; i < stars.size(); ++i) {
        for (size_t j = i + 1; j < stars.size(); ++j) {
            double dx = stars[j].x - stars[i].x;
            double dy = stars[j].y - stars[i].y;
            double r = sqrt(dx * dx + dy * dy);
            double phi = mond_potential(r, stars[j].mass);
            if (!std::isnan(phi) && !std::isinf(phi)) {
                E_p += stars[i].mass * phi;
            }
        }
    }
    E_p *= 2.0;

    double E_total = E_k + E_p;
    if (step == 0) {
        E_total_0 = E_total;
        L_z_0 = L_z;
        cm_x_0 = cm_x;
        cm_y_0 = cm_y;
        cm_vx_0 = cm_vx;
        cm_vy_0 = cm_vy;
        out_quantities << "Step Time_Myr E_kinetic_J E_potential_J E_total_J Delta_E/E0 L_z_kg_m2_s Delta_Lz/Lz0 V_disp_m_s CM_x_m CM_y_m CM_vx_m_s CM_vy_m_s\n";
    }
    double delta_E = E_total_0 != 0 ? (E_total - E_total_0) / std::abs(E_total_0) : 0.0;
    double delta_Lz = L_z_0 != 0 ? (L_z - L_z_0) / std::abs(L_z_0) : 0.0;
    double delta_cm_x = cm_x - cm_x_0;
    double delta_cm_y = cm_y - cm_y_0;
    double delta_cm_vx = cm_vx - cm_vx_0;
    double delta_cm_vy = cm_vy - cm_vy_0;
    double physical_time = step * DT / YEAR / 1e6;
    out_quantities << step << " " << physical_time << " "
                   << E_k << " " << E_p << " " << E_total << " "
                   << delta_E << " " << L_z << " " << delta_Lz << " "
                   << v_disp << " " << delta_cm_x << " " << delta_cm_y << " "
                   << delta_cm_vx << " " << delta_cm_vy << "\n";

    if (step % 2000 == 0) {
        std::cout << "Step: " << step
                  << ", Time: " << physical_time << " Myr"
                  << ", Delta_E/E0: " << delta_E * 100 << "%"
                  << ", Delta_Lz/Lz0: " << delta_Lz * 100 << "%"
                  << ", V_disp: " << v_disp << " m/s"
                  << ", Delta_CM_x: " << delta_cm_x << " m"
                  << ", Delta_CM_vx: " << delta_cm_vx << " m/s\n";
    }
}

// Вычисление кривой вращения
void compute_rotation_curve(const std::vector<Star>& stars, int step) {
    const int N_BINS = 50;
    const double R_MAX = 1e20;
    std::vector<double> v_sum(N_BINS, 0.0);
    std::vector<int> count(N_BINS, 0);

    for (const auto& star : stars) {
        double r = sqrt(star.x * star.x + star.y * star.y);
        if (r > R_MAX) continue;
        int bin = static_cast<int>(r / R_MAX * N_BINS);
        double phi = atan2(star.y, star.x);
        double v_t = -star.vx * sin(phi) + star.vy * cos(phi); // Тангенциальная скорость
        if (std::isnan(v_t) || std::isinf(v_t)) continue;
        v_sum[bin] += fabs(v_t);
        count[bin]++;
    }

    std::ofstream out("rotation_curve_" + std::to_string(step) + ".txt");
    for (int i = 0; i < N_BINS; ++i) {
        double r = (i + 0.5) * R_MAX / N_BINS;
        double v_avg = count[i] > 0 ? v_sum[i] / count[i] : 0.0;
        out << r / KPC << " " << v_avg << "\n";
    }
    out.close();
}

// Вычисление поверхностной плотности
void compute_density_profile(const std::vector<Star>& stars, int step) {
    const int N_BINS = 50;
    const double R_MAX = 1e20;
    std::vector<double> mass_sum(N_BINS, 0.0);
    std::vector<double> area(N_BINS, 0.0);

    for (const auto& star : stars) {
        double r = sqrt(star.x * star.x + star.y * star.y);
        if (r > R_MAX) continue;
        int bin = static_cast<int>(r / R_MAX * N_BINS);
        mass_sum[bin] += star.mass;
    }

    for (int i = 0; i < N_BINS; ++i) {
        double r_inner = i * R_MAX / N_BINS;
        double r_outer = (i + 1) * R_MAX / N_BINS;
        area[i] = M_PI * (r_outer * r_outer - r_inner * r_inner);
    }

    std::ofstream out("density_profile_" + std::to_string(step) + ".txt");
    for (int i = 0; i < N_BINS; ++i) {
        double r = (i + 0.5) * R_MAX / N_BINS;
        double sigma = area[i] > 0 ? mass_sum[i] / area[i] : 0.0;
        out << r / KPC << " " << sigma << "\n";
    }
    out.close();
}

// Сохранение позиций и скоростей
void save_positions_and_velocities(const std::vector<Star>& stars, int step) {
    std::ofstream out("galaxy_step_" + std::to_string(step) + ".txt");
    for (const auto& star : stars) {
        out << star.x << " " << star.y << " " << star.vx << " " << star.vy << "\n";
    }
    out.close();
}

// Вычисление максимального радиуса
double compute_max_radius(const std::vector<Star>& stars) {
    double max_r = 0.0;
    for (const auto& star : stars) {
        double r = sqrt(star.x * star.x + star.y * star.y);
        max_r = std::max(max_r, r);
    }
    return max_r;
}

int main() {
    const int N_STARS = 1000;
    std::vector<Star> stars = generate_galaxy(N_STARS, 12345);

    double L_z_initial = 0.0;
    for (const auto& star : stars) {
        L_z_initial += star.mass * (star.x * star.vy - star.y * star.vx);
    }

    std::ofstream out_quantities("physical_quantities.txt");
    if (!out_quantities.is_open()) {
        std::cerr << "Error: Cannot open physical_quantities.txt" << std::endl;
        return 1;
    }

    std::vector<double> ax, ay;
    double E_total_0 = 0.0, L_z_0 = 0.0;
    double cm_x_0 = 0.0, cm_y_0 = 0.0, cm_vx_0 = 0.0, cm_vy_0 = 0.0;

    std::cout << "Starting relaxation...\n";
    compute_accelerations(stars, ax, ay);
    relax_initial_conditions(stars, ax, ay, L_z_initial);
    correct_center_of_mass(stars);
    std::cout << "Relaxation and center of mass correction completed.\n";

    compute_accelerations(stars, ax, ay);

    for (int step = 0; step < N_STEPS; ++step) {
        update_stars(stars, ax, ay, L_z_initial);

        if (step % 1000 == 0) {
            correct_center_of_mass(stars);
            correct_angular_momentum(stars, L_z_initial);
        }

        double physical_time = step * DT / YEAR / 1e6;
        double max_r = compute_max_radius(stars) / KPC;

        if (step % 2000 == 0) {
            save_positions_and_velocities(stars, step);
            compute_rotation_curve(stars, step);
            compute_density_profile(stars, step);
            compute_physical_quantities(stars, step, out_quantities, E_total_0, L_z_0, cm_x_0, cm_y_0, cm_vx_0, cm_vy_0);
        }
    }

    out_quantities.close();
    return 0;
}