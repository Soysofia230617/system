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
const double MAIN_BELT_AVG_MASS = 4.78e13; // кг
const double KUIPER_BELT_AVG_MASS = 5.0e16; // кг
const double MAIN_BELT_DENSITY = 2.5e3; // кг/м³
const double KUIPER_BELT_DENSITY = 1.0e3; // кг/м³

std::random_device rd;
std::mt19937 gen(rd());

struct Body {
    std::string name;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double eccentricity;
    double inclination;
    double omega;
    double Omega;
    double semi_major_axis;
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
    const int POSITION_SAVE_INTERVAL = 1000;
    const int ENERGY_COMPUTE_INTERVAL = 1000;
    const int PROGRESS_INTERVAL = 1000;

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
            return gm / G;
        }
        if (diameter > 0) {
            double radius = diameter * 1e3 / 2;
            double density = is_kuiper ? KUIPER_BELT_DENSITY : MAIN_BELT_DENSITY;
            return (4.0 / 3.0) * M_PI * radius * radius * radius * density;
        }
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

    void addBody(const std::string& name, double mass, double a_au, double e, double i_deg, double omega_deg, double Omega_deg, double ma_deg) {
        Body body;
        body.name = name;
        body.mass = mass;
        body.eccentricity = e;
        body.inclination = i_deg * M_PI / 180.0;
        body.omega = omega_deg * M_PI / 180.0;
        body.Omega = Omega_deg * M_PI / 180.0;
        body.semi_major_axis = a_au * AU;
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
        if (!body.name.empty()) {
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
            std::getline(file, line);
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

                    double ma = 0.0;
                    double diameter = 0.0;
                    double gm = (gm_str.empty() || gm_str == "null" || gm_str == "\"null\"") ? 0.0 : std::stod(gm_str);

                    double mass = estimateMass(gm, diameter, false);

                    if (a <= 0 || e < 0 || e >= 1 || !std::isfinite(mass)) {
                        std::cerr << "Пропущено тело " << name << " в " << filename << ": некорректные параметры (a=" << a << ", e=" << e << ", mass=" << mass << ")\n";
                        skipped++;
                        continue;
                    }

                    addBody(name, mass, a, e, i, om, w, ma);
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
            std::getline(file, line);
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

                    double ma = 0.0;
                    double diameter = 0.0;
                    double gm = (gm_str.empty() || gm_str == "null" || gm_str == "\"null\"") ? 0.0 : std::stod(gm_str);

                    double mass = estimateMass(gm, diameter, true);

                    if (a <= 0 || e < 0 || e >= 1 || !std::isfinite(mass)) {
                        std::cerr << "Пропущено тело " << name << " в " << filename << ": некорректные параметры (a=" << a << ", e=" << e << ", mass=" << mass << ")\n";
                        skipped++;
                        continue;
                    }

                    addBody(name, mass, a, e, i, om, w, ma);
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

    void shiftToBarycenter() {
        double total_mass = 0.0;
        double bx = 0.0, by = 0.0, bz = 0.0;
        double bvx = 0.0, bvy = 0.0, bvz = 0.0;

        for (const auto& body : bodies) {
            if (!body.is_active) continue;
            total_mass += body.mass;
            bx += body.mass * body.x;
            by += body.mass * body.y;
            bz += body.mass * body.z;
            bvx += body.mass * body.vx;
            bvy += body.mass * body.vy;
            bvz += body.mass * body.vz;
        }

        if (total_mass > 0) {
            bx /= total_mass;
            by /= total_mass;
            bz /= total_mass;
            bvx /= total_mass;
            bvy /= total_mass;
            bvz /= total_mass;
        }

        for (auto& body : bodies) {
            if (!body.is_active) continue;
            body.x -= bx;
            body.y -= by;
            body.z -= bz;
            body.vx -= bvx;
            body.vy -= bvy;
            body.vz -= bvz;
        }

        std::cout << "Барицентр системы перемещён в начало координат: "
                  << "r_b = (" << bx << ", " << by << ", " << bz << "), "
                  << "v_b = (" << bvx << ", " << bvy << ", " << bvz << ")\n";
    }

    void computeBarycenter(double& bx, double& by, double& bz, double& bvx, double& bvy, double& bvz) {
        double total_mass = 0.0;
        bx = by = bz = bvx = bvy = bvz = 0.0;

        for (const auto& body : bodies) {
            if (!body.is_active) continue;
            total_mass += body.mass;
            bx += body.mass * body.x;
            by += body.mass * body.y;
            bz += body.mass * body.z;
            bvx += body.mass * body.vx;
            bvy += body.mass * body.vy;
            bvz += body.mass * body.vz;
        }

        if (total_mass > 0) {
            bx /= total_mass;
            by /= total_mass;
            bz /= total_mass;
            bvx /= total_mass;
            bvy /= total_mass;
            bvz /= total_mass;
        }
    }

    void computeAngularMomentum(double& Lx, double& Ly, double& Lz) {
        Lx = Ly = Lz = 0.0;
        for (const auto& body : bodies) {
            if (!body.is_active) continue;
            Lx += body.mass * (body.y * body.vz - body.z * body.vy);
            Ly += body.mass * (body.z * body.vx - body.x * body.vz);
            Lz += body.mass * (body.x * body.vy - body.y * body.vx);
        }
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
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                if (!bodies[j].is_active) continue;
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double r2 = dx * dx + dy * dy + dz * dz;
                double r = std::sqrt(r2);
                double coef = G / (r2 * r);
                double force_i = coef * bodies[j].mass;
                double force_j = coef * bodies[i].mass;
                ax[i] += force_i * dx;
                ay[i] += force_i * dy;
                az[i] += force_i * dz;
                ax[j] -= force_j * dx;
                ay[j] -= force_j * dy;
                az[j] -= force_j * dz;
            }
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

    void simulate() {
        std::ofstream energy_file("energy.txt");
        std::ofstream traj_file("trajectories.csv");
        std::ofstream inv_file("invariants.txt");
        traj_file << "step,name,x_au,y_au,z_au\n";
        inv_file << "step,time_days,energy,delta_e,bx,by,bz,bvx,bvy,bvz,Lx,Ly,Lz\n";

        shiftToBarycenter();
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

            if (step % POSITION_SAVE_INTERVAL == 0) {
                for (const auto& body : bodies) {
                    if (body.is_active) {
                        traj_file << step << "," << body.name << ","
                                  << body.x / AU << "," << body.y / AU << "," << body.z / AU << "\n";
                    }
                }
            }

            if (step % ENERGY_COMPUTE_INTERVAL == 0) {
                total_energy = computeEnergy();
                double delta_e = (std::abs(initial_energy) > 1e-10) ? std::abs(total_energy - initial_energy) / std::abs(initial_energy) : 0.0;
                max_delta_e = std::max(max_delta_e, delta_e);

                double bx, by, bz, bvx, bvy, bvz;
                computeBarycenter(bx, by, bz, bvx, bvy, bvz);
                double Lx, Ly, Lz;
                computeAngularMomentum(Lx, Ly, Lz);

                double time_days = step * dt / DAY;
                inv_file << step << "," << std::fixed << std::setprecision(3) << time_days << ","
                         << total_energy << "," << delta_e << ","
                         << bx << "," << by << "," << bz << ","
                         << bvx << "," << bvy << "," << bvz << ","
                         << Lx << "," << Ly << "," << Lz << "\n";
            }

            if (step % PROGRESS_INTERVAL == 0) {
                double time_years = step * dt / YEAR;
                std::cout << "Прогресс: шаг " << step << " из " << steps << ", время = " << std::fixed << std::setprecision(2)
                          << time_years << " лет\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Время выполнения симуляции: " << duration.count() << " секунд\n";

        energy_file.close();
        traj_file.close();
        inv_file.close();
    }

    void printResults(double total_time_years) {
        std::cout << "Результаты моделирования:\n";
        std::cout << "Общее время моделирования: " << std::fixed << std::setprecision(2)
                  << total_time_years << " лет\n";
        std::cout << "Шаг интегрирования: " << std::fixed << std::setprecision(4)
                  << dt / DAY << " дней\n";
        std::cout << "Максимальное отклонение энергии: " << std::scientific << max_delta_e << "\n";
    }
};

int main() {
    SolarSystem system(0.01, 20.0);

    system.addBody("Sun", M_SUN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    system.addBody("Mercury", 3.30e23, 0.387, 0.2056, 7.00, 29.12, 48.33, 0.0);
    system.addBody("Venus", 4.87e24, 0.723, 0.0068, 3.39, 54.88, 76.68, 0.0);
    system.addBody("Earth", 5.97e24, 1.000, 0.0167, 0.00, 114.21, 348.74, 0.0);
    system.addBody("Mars", 6.42e23, 1.524, 0.0934, 1.85, 49.56, 286.50, 0.0);
    system.addBody("Jupiter", 1.90e27, 5.204, 0.0489, 1.30, 100.46, 275.07, 0.0);
    system.addBody("Saturn", 5.68e26, 9.582, 0.0565, 2.49, 113.67, 339.39, 0.0);
    system.addBody("Uranus", 8.68e25, 19.218, 0.0463, 0.77, 74.00, 96.54, 0.0);
    system.addBody("Neptune", 1.02e26, 30.070, 0.0095, 1.77, 131.78, 276.34, 0.0);
    system.addBody("Ceres", 9.39e20, 2.77, 0.0758, 10.59, 80.33, 73.12, 0.0);
    system.addBody("Vesta", 2.59e20, 2.36, 0.0887, 7.14, 103.85, 150.73, 0.0);
    system.addBody("Pallas", 2.11e20, 2.77, 0.2313, 34.84, 173.09, 310.05, 0.0);
    system.addBody("Hygiea", 8.67e19, 3.14, 0.1126, 3.84, 283.20, 312.32, 0.0);
    system.addBody("Pluto", 1.31e22, 39.482, 0.2488, 17.14, 110.30, 113.76, 0.0);
    system.addBody("Eris", 1.66e22, 67.8, 0.4361, 44.04, 35.95, 150.98, 0.0);
    system.addBody("Haumea", 4.01e21, 43.1, 0.1913, 28.19, 121.79, 239.08, 0.0);
    system.addBody("Makemake", 3.1e21, 45.8, 0.1610, 29.01, 79.36, 297.24, 0.0);

    std::vector<std::string> main_belt_files = {
            "main_belt_test.csv"
    };
    std::vector<std::string> kuiper_belt_files = {
            "kuiper_belt_1.csv"
    };

    system.loadMainBelt(main_belt_files);
    system.loadKuiperBelt(kuiper_belt_files);

    system.simulate();
    system.printResults(20.0);

    return 0;
}
