#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

const double G = 6.67430e-11;
const double M_SUN = 1.989e30;
const double AU = 1.496e11;
const double DAY = 86400.0;
const double YEAR = 365.25 * DAY;
const double m_PI = 3.14159265358979323846;

const double MAIN_BELT_TOTAL_MASS = 2.39e20; // кг
const double KUIPER_BELT_TOTAL_MASS = 1.0e22; // кг
const double MAIN_BELT_AVG_MASS = 4.78e13; // кг, для мелких астероидов
const double KUIPER_BELT_AVG_MASS = 5.0e16; // кг, для мелких объектов
const double MAIN_BELT_DENSITY = 2.5e3; // кг/м³
const double KUIPER_BELT_DENSITY = 1.0e3; // кг/м³

std::random_device rd;
std::mt19937 gen(rd());

struct Body {
    std::string name;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double period;
    double min_omega, max_omega;
    double expected_period;
    double eccentricity;
    double inclination;
    double omega;
    double Omega;
    double semi_major_axis;
    std::vector<double> positions_x, positions_y;
    std::vector<double> crossing_times;
    bool period_computed;
    bool is_random_asteroid;
    bool is_active;
};

class SolarSystem {
private:
    std::vector<Body> bodies;
    double dt;
    int steps;
    double total_energy;
    double initial_energy;
    double max_delta_e;
    const int POSITION_SAVE_INTERVAL = 2000;
    const int ENERGY_COMPUTE_INTERVAL = 50000;
    const int COLLISION_CHECK_INTERVAL = 5000;

public:
    SolarSystem(double time_step_days, double total_time_years) {
        dt = time_step_days * DAY;
        steps = static_cast<int>(total_time_years * YEAR / dt);
        total_energy = 0.0;
        initial_energy = 0.0;
        max_delta_e = 0.0;
#ifdef _OPENMP
        std::cout << "OpenMP включен, число потоков: " << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP отключен, используется однопоточная версия\n";
#endif
    }

    double estimateMass(double gm, double diameter, bool is_kuiper) {
        if (gm > 0) {
            return gm / G; // Масса из GM
        }
        if (diameter > 0) {
            double radius = diameter * 1e3 / 2; // Диаметр в км -> радиус в м
            double density = is_kuiper ? KUIPER_BELT_DENSITY : MAIN_BELT_DENSITY;
            return (4.0 / 3.0) * M_PI * radius * radius * radius * density;
        }
        // Средняя масса для объектов без GM и диаметра
        return is_kuiper ? KUIPER_BELT_AVG_MASS : MAIN_BELT_AVG_MASS;
    }

    void setInitialConditions(Body& body, double mean_anomaly_rad) {
        double a = body.semi_major_axis;
        double e = body.eccentricity;
        double i = body.inclination;
        double omega = body.omega;
        double Omega = body.Omega;

        double E = mean_anomaly_rad;
        for (int iter = 0; iter < 100; ++iter) {
            double delta_E = (E - e * std::sin(E) - mean_anomaly_rad) / (1 - e * std::cos(E));
            E -= delta_E;
            if (std::abs(delta_E) < 1e-10) break;
        }

        double true_anomaly = 2 * std::atan2(std::sqrt(1 + e) * std::sin(E / 2), std::sqrt(1 - e) * std::cos(E / 2));
        double r = a * (1 - e * std::cos(E));

        double x_orb = r * std::cos(true_anomaly);
        double y_orb = r * std::sin(true_anomaly);

        body.x = (std::cos(omega) * std::cos(Omega) - std::sin(omega) * std::sin(Omega) * std::cos(i)) * x_orb +
                 (-std::sin(omega) * std::cos(Omega) - std::cos(omega) * std::sin(Omega) * std::cos(i)) * y_orb;
        body.y = (std::cos(omega) * std::sin(Omega) + std::sin(omega) * std::cos(Omega) * std::cos(i)) * x_orb +
                 (-std::sin(omega) * std::sin(Omega) + std::cos(omega) * std::cos(Omega) * std::cos(i)) * y_orb;
        body.z = (std::sin(omega) * std::sin(i)) * x_orb + (std::cos(omega) * std::sin(i)) * y_orb;

        double mu = G * M_SUN;
        double p = a * (1 - e * e);
        double v_x_orb = -std::sqrt(mu / p) * std::sin(true_anomaly);
        double v_y_orb = std::sqrt(mu / p) * (e + std::cos(true_anomaly));

        body.vx = (std::cos(omega) * std::cos(Omega) - std::sin(omega) * std::sin(Omega) * std::cos(i)) * v_x_orb +
                  (-std::sin(omega) * std::cos(Omega) - std::cos(omega) * std::sin(Omega) * std::cos(i)) * v_y_orb;
        body.vy = (std::cos(omega) * std::sin(Omega) + std::sin(omega) * std::cos(Omega) * std::cos(i)) * v_x_orb +
                  (-std::sin(omega) * std::sin(Omega) + std::cos(omega) * std::cos(Omega) * std::cos(i)) * v_y_orb;
        body.vz = (std::sin(omega) * std::sin(i)) * v_x_orb + (std::cos(omega) * std::sin(i)) * v_y_orb;
    }

    void addBody(const std::string& name, double mass, double a_au, double e, double i_deg, double omega_deg, double Omega_deg, double ma_deg, double expected_period_years, bool is_random = false) {
        Body body;
        body.name = name;
        body.mass = mass;
        body.eccentricity = e;
        body.inclination = i_deg * M_PI / 180.0;
        body.omega = omega_deg * M_PI / 180.0;
        body.Omega = Omega_deg * M_PI / 180.0;
        body.semi_major_axis = a_au * AU;
        body.expected_period = expected_period_years;
        body.period = 0.0;
        body.min_omega = 1e10;
        body.max_omega = -1e10;
        body.period_computed = false;
        body.is_random_asteroid = is_random;
        body.is_active = true;

        if (a_au == 0.0) {
            body.x = 0.0;
            body.y = 0.0;
            body.z = 0.0;
            body.vx = 0.0;
            body.vy = 0.0;
            body.vz = 0.0;
        } else {
            double mean_anomaly_rad = ma_deg * M_PI / 180.0;
            setInitialConditions(body, mean_anomaly_rad);
        }

        bodies.push_back(body);
        if (!is_random && !body.name.empty()) {
            std::cout << "Добавлено тело: " << name << ", масса = " << mass << " кг, эксцентриситет = " << e
                      << ", наклонение = " << i_deg << " град\n";
        }
    }

    void loadMainBelt(const std::vector<std::string>& filenames) {
        int total_loaded = 0, total_skipped = 0;
        double mass_sum = 0.0;

        for (const auto& filename : filenames) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Ошибка: не удалось открыть файл главного пояса " << filename << "\n";
                continue;
            }
            std::string line;
            std::getline(file, line); // Пропустить заголовок
            int loaded = 0, skipped = 0;
            while (std::getline(file, line)) {
                try {
                    std::stringstream ss(line);
                    std::string name, gm_str, token;
                    std::vector<std::string> tokens;
                    while (std::getline(ss, token, ',')) {
                        tokens.push_back(token);
                    }
                    if (tokens.size() < 8) {
                        std::cerr << "Пропущена строка в " << filename << " (недостаточно полей): " << line << "\n";
                        skipped++;
                        continue;
                    }

                    // Ожидаемый формат: full_name,epoch_mjd,e,a,i,om,GM,w
                    name = tokens[0];
                    if (!name.empty() && name.front() == '"' && name.back() == '"') {
                        name = name.substr(1, name.size() - 2);
                    }
                    name.erase(0, name.find_first_not_of(" \t"));
                    name.erase(name.find_last_not_of(" \t") + 1);

                    double epoch_mjd = std::stod(tokens[1]);
                    double e = std::stod(tokens[2]);
                    double a = std::stod(tokens[3]);
                    double i = std::stod(tokens[4]);
                    double om = std::stod(tokens[5]);
                    gm_str = tokens[6];
                    double w = std::stod(tokens[7]);

                    // Преобразование epoch_mjd в epoch (JD)
                    double epoch = epoch_mjd + 2400000.5;
                    // Средняя аномалия и диаметр отсутствуют
                    double ma = 0.0;
                    double diameter = 0.0;
                    // Обработка GM
                    double gm = (gm_str.empty() || gm_str == "null" || gm_str == "\"null\"") ? 0.0 : std::stod(gm_str);

                    double mass = estimateMass(gm, diameter, false); // is_kuiper = false
                    double expected_period = std::sqrt(a * a * a);

                    if (a <= 0 || e < 0 || e >= 1 || !std::isfinite(mass)) {
                        std::cerr << "Пропущено тело " << name << " в " << filename << ": некорректные параметры (a=" << a << ", e=" << e << ", mass=" << mass << ")\n";
                        skipped++;
                        continue;
                    }

                    addBody(name, mass, a, e, i, om, w, ma, expected_period, true);
                    mass_sum += mass;
                    loaded++;
                } catch (const std::exception& ex) {
                    std::cerr << "Ошибка парсинга строки в " << filename << ": " << line << " (" << ex.what() << ")\n";
                    skipped++;
                }
            }
            file.close();
            total_loaded += loaded;
            total_skipped += skipped;
            std::cout << "Главный пояс, файл " << filename << ": загружено " << loaded << " тел, пропущено " << skipped << "\n";
        }
        std::cout << "Главный пояс, всего: загружено " << total_loaded << " тел, пропущено " << total_skipped << "\n";
        std::cout << "Суммарная масса главного пояса: " << mass_sum << " кг (ожидается ~" << MAIN_BELT_TOTAL_MASS << " кг)\n";
    }

    void loadKuiperBelt(const std::vector<std::string>& filenames) {
        int total_loaded = 0, total_skipped = 0;
        double mass_sum = 0.0;

        for (const auto& filename : filenames) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Ошибка: не удалось открыть файл пояса Койпера " << filename << "\n";
                continue;
            }
            std::string line;
            std::getline(file, line); // Пропустить заголовок
            int loaded = 0, skipped = 0;
            while (std::getline(file, line)) {
                try {
                    std::stringstream ss(line);
                    std::string name, gm_str, token;
                    std::vector<std::string> tokens;
                    while (std::getline(ss, token, ',')) {
                        tokens.push_back(token);
                    }
                    if (tokens.size() < 8) {
                        std::cerr << "Пропущена строка в " << filename << " (недостаточно полей): " << line << "\n";
                        skipped++;
                        continue;
                    }

                    // Ожидаемый формат: full_name,epoch_mjd,e,a,i,om,GM,w
                    name = tokens[0];
                    if (!name.empty() && name.front() == '"' && name.back() == '"') {
                        name = name.substr(1, name.size() - 2);
                    }
                    // Удаляем лишние пробелы из имени
                    name.erase(0, name.find_first_not_of(" \t"));
                    name.erase(name.find_last_not_of(" \t") + 1);

                    double epoch_mjd = std::stod(tokens[1]);
                    double e = std::stod(tokens[2]);
                    double a = std::stod(tokens[3]);
                    double i = std::stod(tokens[4]);
                    double om = std::stod(tokens[5]);
                    gm_str = tokens[6];
                    double w = std::stod(tokens[7]);

                    // Преобразование epoch_mjd в epoch (JD)
                    double epoch = epoch_mjd + 2400000.5;
                    // Средняя аномалия и диаметр отсутствуют
                    double ma = 0.0;
                    double diameter = 0.0;
                    // Обработка GM
                    double gm = (gm_str.empty() || gm_str == "null" || gm_str == "\"null\"") ? 0.0 : std::stod(gm_str);

                    double mass = estimateMass(gm, diameter, true); // is_kuiper = true
                    double expected_period = std::sqrt(a * a * a);

                    if (a <= 0 || e < 0 || e >= 1 || !std::isfinite(mass)) {
                        std::cerr << "Пропущено тело " << name << " в " << filename << ": некорректные параметры (a=" << a << ", e=" << e << ", mass=" << mass << ")\n";
                        skipped++;
                        continue;
                    }

                    addBody(name, mass, a, e, i, om, w, ma, expected_period, true);
                    mass_sum += mass;
                    loaded++;
                } catch (const std::exception& ex) {
                    std::cerr << "Ошибка парсинга строки в " << filename << ": " << line << " (" << ex.what() << ")\n";
                    skipped++;
                }
            }
            file.close();
            total_loaded += loaded;
            total_skipped += skipped;
            std::cout << "Пояс Койпера, файл " << filename << ": загружено " << loaded << " тел, пропущено " << skipped << "\n";
        }
        std::cout << "Пояс Койпера, всего: загружено " << total_loaded << " тел, пропущено " << total_skipped << "\n";
        std::cout << "Суммарная масса пояса Койпера: " << mass_sum << " кг (ожидается ~" << KUIPER_BELT_TOTAL_MASS << " кг)\n";
    }

    void computeAccelerations(std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az) {
        ax.assign(bodies.size(), 0.0);
        ay.assign(bodies.size(), 0.0);
        az.assign(bodies.size(), 0.0);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!bodies[i].is_active) continue;
            double ax_i = 0.0, ay_i = 0.0, az_i = 0.0;
            for (size_t j = 0; j < bodies.size(); ++j) {
                if (i == j || !bodies[j].is_active) continue;
                if (bodies[i].is_random_asteroid && bodies[j].is_random_asteroid) continue;
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < 1e-20) continue;
                double r = std::sqrt(r2);
                double force = G * bodies[j].mass / (r2 * r);
                ax_i += force * dx;
                ay_i += force * dy;
                az_i += force * dz;
            }
            ax[i] = ax_i;
            ay[i] = ay_i;
            az[i] = az_i;
        }
    }

    void checkCollisions() {
        static int collision_count = 0;
        static std::vector<std::string> collision_log;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!bodies[i].is_active || !bodies[i].is_random_asteroid) continue;
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                if (!bodies[j].is_active || !bodies[j].is_random_asteroid) continue;
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e6) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    {
                        if (bodies[i].is_active && bodies[j].is_active) {
                            collision_log.push_back("Столкновение: " + bodies[i].name + " и " + bodies[j].name + "\n");
                            bodies[i].is_active = false;
                            collision_count++;
                        }
                    }
                }
            }
        }

        if (collision_count >= 500) {
            for (const auto& log : collision_log) {
                std::cout << log;
            }
            collision_log.clear();
            collision_count = 0;
        }
    }

    double computeEnergy() {
        double kinetic = 0.0, potential = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:kinetic,potential)
#endif
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!bodies[i].is_active) continue;
            double v2 = bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy + bodies[i].vz * bodies[i].vz;
            if (!std::isfinite(v2)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                std::cerr << "Ошибка: некорректная скорость для " << bodies[i].name << "\n";
                continue;
            }
            kinetic += 0.5 * bodies[i].mass * v2;

            for (size_t j = i + 1; j < bodies.size(); ++j) {
                if (!bodies[j].is_active) continue;
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e6) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    std::cerr << "Ошибка: расстояние между " << bodies[i].name << " и " << bodies[j].name << " слишком мало\n";
                    continue;
                }
                double pot = -G * bodies[i].mass * bodies[j].mass / r;
                if (!std::isfinite(pot)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    std::cerr << "Ошибка: некорректная потенциальная энергия между " << bodies[i].name << " и " << bodies[j].name << "\n";
                    continue;
                }
                potential += pot;
            }
        }
        double energy = kinetic + potential;
        if (!std::isfinite(energy)) {
            std::cerr << "Ошибка: общая энергия системы некорректна\n";
        }
        return energy;
    }

    void computeAngularVelocity(int step) {
        for (size_t i = 1; i < bodies.size(); ++i) {
            if (bodies[i].period_computed || bodies[i].is_random_asteroid || !bodies[i].is_active) continue;

            double r = std::sqrt(bodies[i].x * bodies[i].x + bodies[i].y * bodies[i].y);
            double v = std::sqrt(bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy);
            if (r < 1e6) continue;
            double omega = v / r;
            bodies[i].min_omega = std::min(bodies[i].min_omega, omega);
            bodies[i].max_omega = std::max(bodies[i].max_omega, omega);

            if (step % POSITION_SAVE_INTERVAL == 0) {
                bodies[i].positions_x.push_back(bodies[i].x);
                bodies[i].positions_y.push_back(bodies[i].y);
            }
        }
    }

    void computeOrbitalPeriods(int step) {
        for (size_t i = 1; i < bodies.size(); ++i) {
            if (bodies[i].period_computed || bodies[i].is_random_asteroid || !bodies[i].is_active) continue;

            if (bodies[i].positions_x.size() < 2) continue;

            int crossings = bodies[i].crossing_times.size();
            double last_x = bodies[i].positions_x[bodies[i].positions_x.size() - 2];
            double current_x = bodies[i].positions_x.back();

            if ((last_x < 0 && current_x >= 0) || (last_x > 0 && current_x <= 0)) {
                crossings++;
                double time = (step / POSITION_SAVE_INTERVAL) * (dt * POSITION_SAVE_INTERVAL) / YEAR;
                bodies[i].crossing_times.push_back(time);
            }

            std::vector<double> periods;
            for (size_t k = 1; k < bodies[i].crossing_times.size(); k += 2) {
                if (k + 1 < bodies[i].crossing_times.size()) {
                    double period = (bodies[i].crossing_times[k + 1] - bodies[i].crossing_times[k - 1]);
                    periods.push_back(period);
                }
            }

            if (periods.size() >= 1) {
                double period_sum = 0.0;
                for (double p : periods) period_sum += p;
                bodies[i].period = period_sum / periods.size();
                if (crossings >= 6) {
                    bodies[i].period_computed = true;
                    std::cout << "Для " << bodies[i].name << " найдено " << crossings << " пересечений, период: " << bodies[i].period << " лет\n";
                }
            }
        }
    }

    void simulate() {
        std::ofstream energy_file("energy.txt");
        std::ofstream traj_file("trajectories.csv");
        traj_file << "step,name,x_au,y_au,z_au\n";

        initial_energy = computeEnergy();
        total_energy = initial_energy;
        std::cout << "Начальная энергия: " << initial_energy << "\n";

        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; ++step) {
            std::vector<double> ax(bodies.size()), ay(bodies.size()), az(bodies.size());
            computeAccelerations(ax, ay, az);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < bodies.size(); ++i) {
                if (!bodies[i].is_active) continue;
                double new_x = bodies[i].x + bodies[i].vx * dt + 0.5 * ax[i] * dt * dt;
                double new_y = bodies[i].y + bodies[i].vy * dt + 0.5 * ay[i] * dt * dt;
                double new_z = bodies[i].z + bodies[i].vz * dt + 0.5 * az[i] * dt * dt;
                bodies[i].x = new_x;
                bodies[i].y = new_y;
                bodies[i].z = new_z;
            }

            std::vector<double> ax_new(bodies.size()), ay_new(bodies.size()), az_new(bodies.size());
            computeAccelerations(ax_new, ay_new, az_new);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < bodies.size(); ++i) {
                if (!bodies[i].is_active) continue;
                double new_vx = bodies[i].vx + 0.5 * (ax[i] + ax_new[i]) * dt;
                double new_vy = bodies[i].vy + 0.5 * (ay[i] + ay_new[i]) * dt;
                double new_vz = bodies[i].vz + 0.5 * (az[i] + az_new[i]) * dt;
                bodies[i].vx = new_vx;
                bodies[i].vy = new_vy;
                bodies[i].vz = new_vz;
            }

            if (step % COLLISION_CHECK_INTERVAL == 0) {
                checkCollisions();
            }

            computeAngularVelocity(step);
            if (step % POSITION_SAVE_INTERVAL == 0) {
                computeOrbitalPeriods(step);
                for (const auto& body : bodies) {
                    if (body.is_active && !body.is_random_asteroid) {
                        traj_file << step << "," << body.name << ","
                                  << body.x / AU << "," << body.y / AU << "," << body.z / AU << "\n";
                    }
                }
            }

            if (step % ENERGY_COMPUTE_INTERVAL == 0) {
                total_energy = computeEnergy();
                double delta_e = (std::abs(initial_energy) > 1e-10) ? std::abs(total_energy - initial_energy) / std::abs(initial_energy) : 0.0;
                max_delta_e = std::max(max_delta_e, delta_e);

                double time_days = step * dt / DAY;
                double log_delta_e = (delta_e != 0) ? std::log10(delta_e) : -20.0;
                energy_file << "Шаг " << step << ": Время = " << std::fixed << std::setprecision(3)
                            << time_days << " дней, Энергия = " << total_energy
                            << ", Отклонение = " << delta_e
                            << ", log10|ΔE/E₀| = " << log_delta_e << "\n";
            }

            bool all_short_periods_computed = true;
            for (size_t i = 1; i < bodies.size(); ++i) {
                if (bodies[i].expected_period <= 5.0 && !bodies[i].period_computed && bodies[i].is_active) {
                    all_short_periods_computed = false;
                    break;
                }
            }
            if (all_short_periods_computed) {
                std::cout << "Все периоды для тел с ожидаемым периодом <= 5 лет вычислены, симуляция завершена на шаге " << step << "\n";
                break;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Время выполнения симуляции: " << duration.count() << " секунд\n";

        energy_file.close();
        traj_file.close();
    }

    void printResults(double total_time_years) {
        std::cout << "Результаты моделирования:\n";
        std::cout << "Общее время моделирования: " << std::fixed << std::setprecision(2)
                  << total_time_years << " лет\n";
        std::cout << "Шаг интегрирования: " << std::fixed << std::setprecision(4)
                  << dt / DAY << " дней\n";
        std::cout << "Максимальное отклонение энергии: " << std::scientific << max_delta_e << "\n\n";

        std::cout << "Орбитальные периоды (годы):\n";
        for (size_t i = 1; i < bodies.size(); ++i) {
            if (bodies[i].is_random_asteroid || !bodies[i].is_active) continue;
            if (bodies[i].period == 0) {
                std::cout << bodies[i].name << ": период не вычислен\n";
                continue;
            }
            double error = std::abs(bodies[i].period - bodies[i].expected_period) / bodies[i].expected_period * 100.0;
            std::cout << bodies[i].name << ": " << std::fixed << std::setprecision(2)
                      << bodies[i].period << " (ожидается " << bodies[i].expected_period
                      << ", ошибка " << error << "%)\n";
        }

        std::cout << "\nУгловые скорости (рад/с):\n";
        for (size_t i = 1; i < bodies.size(); ++i) {
            if (bodies[i].is_random_asteroid || !bodies[i].is_active) continue;
            std::cout << bodies[i].name << ": min=" << std::scientific
                      << bodies[i].min_omega << ", max=" << bodies[i].max_omega << "\n";
        }
    }
};

int main() {
    SolarSystem system(0.01, 270.0);

    // Добавление основных тел
    system.addBody("Sun", M_SUN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    system.addBody("Mercury", 3.30e23, 0.387, 0.2056, 7.00, 29.12, 48.33, 0.0, 0.24);
    system.addBody("Venus", 4.87e24, 0.723, 0.0068, 3.39, 54.88, 76.68, 0.0, 0.62);
    system.addBody("Earth", 5.97e24, 1.000, 0.0167, 0.00, 114.21, 348.74, 0.0, 1.00);
    system.addBody("Mars", 6.42e23, 1.524, 0.0934, 1.85, 49.56, 286.50, 0.0, 1.88);
    system.addBody("Jupiter", 1.90e27, 5.204, 0.0489, 1.30, 100.46, 275.07, 0.0, 11.86);
    system.addBody("Saturn", 5.68e26, 9.582, 0.0565, 2.49, 113.67, 339.39, 0.0, 29.46);
    system.addBody("Uranus", 8.68e25, 19.218, 0.0463, 0.77, 74.00, 96.54, 0.0, 84.01);
    system.addBody("Neptune", 1.02e26, 30.070, 0.0095, 1.77, 131.78, 276.34, 0.0, 164.80);
    system.addBody("Ceres", 9.39e20, 2.77, 0.0758, 10.59, 80.33, 73.12, 0.0, 4.60);
    system.addBody("Vesta", 2.59e20, 2.36, 0.0887, 7.14, 103.85, 150.73, 0.0, 3.63);
    system.addBody("Pallas", 2.11e20, 2.77, 0.2313, 34.84, 173.09, 310.05, 0.0, 4.60);
    system.addBody("Hygiea", 8.67e19, 3.14, 0.1126, 3.84, 283.20, 312.32, 0.0, 5.56);
    system.addBody("Asteroid_Alpha", 1.0e20, 7.4, 0.05, 5.0, 0.0, 0.0, 0.0, 20.12);
    system.addBody("Asteroid_Beta", 5.0e19, 7.5, 0.05, 5.0, 0.0, 0.0, 0.0, 20.52);
    system.addBody("Pluto", 1.31e22, 39.482, 0.2488, 17.14, 110.30, 113.76, 0.0, 248.10);
    system.addBody("Eris", 1.66e22, 67.8, 0.4361, 44.04, 35.95, 150.98, 0.0, 558.00);
    system.addBody("Haumea", 4.01e21, 43.1, 0.1913, 28.19, 121.79, 239.08, 0.0, 283.00);
    system.addBody("Makemake", 3.1e21, 45.8, 0.1610, 29.01, 79.36, 297.24, 0.0, 309.00);

    // Загрузка астероидов из нескольких файлов
    std::vector<std::string> main_belt_files = {
            "main_belt_1.csv"
    };
    std::vector<std::string> kuiper_belt_files = {
            "kuiper_belt_1.csv",
            "kuiper_belt_2.csv",
            "kuiper_belt_3.csv",
            "kuiper_belt_4.csv",
            "kuiper_belt_5.csv"
    };

    system.loadMainBelt(main_belt_files);
    //system.loadKuiperBelt(kuiper_belt_files);

    system.simulate();
    system.printResults(270.0);

    return 0;
}
